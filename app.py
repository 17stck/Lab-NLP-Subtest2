"""
🌱 ผู้ช่วยปลูกผักสวนครัวในบ้าน — RAG Chatbot
Document Loading & Chunking -> Embedding + FAISS -> Prompt Engineering -> Gemini -> Streamlit chat
"""
import glob
import html
import json
import os
import re
import tempfile
import time
import uuid

import faiss
import numpy as np
import streamlit as st
from google import genai
from pythainlp.util import normalize
from sentence_transformers import SentenceTransformer

# ----------------------------- การตั้งค่า -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBED_MODEL = "intfloat/multilingual-e5-small"  # โมเดลเล็ก รองรับไทย/อังกฤษ
LLM_MODEL = "gemini-3.8-flash"
CHUNK_SIZE = 450       # จำนวนตัวอักษรสูงสุดต่อ chunk
CHUNK_OVERLAP = 120    # หน่วยข้อความท้าย chunk ที่ยกไปซ้อนกับ chunk ถัดไป
HISTORY_TURNS = 6      # จำนวนข้อความย้อนหลังที่ส่งให้ LLM

EXAMPLE_QUESTIONS = [
    "ปลูกผักในกระถางต้องการแดดวันละกี่ชั่วโมง?",
    "ผักบุ้งจีนใช้เวลากี่วันถึงเก็บเกี่ยวได้?",
    "ใบกะเพราเหลืองเกิดจากอะไร?",
    "ทำปุ๋ยหมักจากเศษอาหารอย่างไร?",
    "ราคาเมล็ดพันธุ์คะน้าซองละเท่าไร?",
]

EXAMPLE_QUESTIONS_EN = [
    "How many hours of sunlight do vegetables in pots need each day?",
    "How long does Chinese water spinach take to harvest?",
    "Why are my holy basil leaves turning yellow?",
    "How can I make compost from kitchen scraps?",
    "How much does a packet of Chinese kale seeds cost?",
]

SYSTEM_PROMPT = """You are a home vegetable gardening assistant. Answer using only the supplied reference documents.

Rules:
1. Do not use outside knowledge or invent facts.
2. Add source-number citations such as [1] or [2] to every factual statement.
3. If the documents do not answer the question, say so in the requested language and do not add citations.
4. Use conversation history only to understand follow-up questions; factual claims must still come from the documents.
5. Keep the answer concise, clear, and step-by-step when appropriate.
"""


# ----------------------------- โหลด + ทำความสะอาด + แบ่ง chunk -----------------------------
def clean_text(text: str) -> str:
    text = normalize(text)
    text = text.replace("\u200b", "").replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_language(text: str) -> str:
    """Choose Thai or English based on the script used most in the latest question."""
    thai_letters = len(re.findall(r"[\u0e00-\u0e7f]", text))
    latin_letters = len(re.findall(r"[A-Za-z]", text))
    return "th" if thai_letters >= latin_letters else "en"


def split_long(unit: str, size: int) -> list[str]:
    """Split long paragraphs into sentences, then into bounded chunks."""
    try:
        from pythainlp.tokenize import sent_tokenize
        parts = sent_tokenize(unit, engine="crfcut")
    except Exception:
        parts = re.split(r"(?<=[.!?])\s+", unit)
    out = []
    for p in parts:
        p = p.strip()
        while len(p) > size:
            out.append(p[:size])
            p = p[size:]
        if p:
            out.append(p)
    return out


def chunk_document(text: str) -> tuple[str, list[str]]:
    """คืน (หัวข้อเอกสาร, รายการ chunk)"""
    text = clean_text(text)
    lines = text.split("\n", 1)
    title = lines[0].strip()
    body = lines[1] if len(lines) > 1 else ""

    units: list[str] = []
    for para in re.split(r"\n\s*\n", body):
        para = re.sub(r"\s+", " ", para).strip()
        if not para:
            continue
        units.extend([para] if len(para) <= CHUNK_SIZE else split_long(para, CHUNK_SIZE))

    chunks, current = [], ""
    for u in units:
        if current and len(current) + len(u) + 1 > CHUNK_SIZE:
            chunks.append(current)
            carry_size = min(CHUNK_OVERLAP, max(0, CHUNK_SIZE - len(u) - 1))
            carry = current[-carry_size:] if carry_size else ""
            current = (carry + " " + u).strip()
        else:
            current = (current + " " + u).strip()
    if current:
        chunks.append(current)
    return title, chunks


@st.cache_resource
def build_index():
    """โหลดโมเดล + เอกสาร + FAISS index เพียงครั้งเดียว"""
    model = SentenceTransformer(EMBED_MODEL)
    records = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        title, chunks = chunk_document(raw)
        for i, c in enumerate(chunks, 1):
            records.append({
                "source": os.path.basename(path),
                "title": title,
                "chunk_id": i,
                "text": c,
            })
    if not records:
        raise RuntimeError(tr(
            "ไม่พบไฟล์ .txt ในโฟลเดอร์ data/",
            "No .txt files were found in the data/ folder.",
        ))

    # e5 ต้องใส่ prefix "passage: " ; แนบหัวข้อเพื่อช่วยให้ค้นหาแม่นขึ้น
    passages = [f"passage: {r['title']} — {r['text']}" for r in records]
    emb = model.encode(passages, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    emb = np.asarray(emb, dtype="float32")
    index = faiss.IndexFlatIP(emb.shape[1])  # inner product บนเวกเตอร์ normalized = cosine
    index.add(emb)
    return model, index, records


# ----------------------------- Retrieval -----------------------------
def retrieve(query: str, k: int, min_score: float, allowed: list[str] | None = None):
    """ค้นหา Top-K chunk; ถ้ากำหนด allowed จะค้นเฉพาะเอกสารที่โปรเจกต์เลือกไว้"""
    model, index, records = build_index()
    q = model.encode([f"query: {query}"], normalize_embeddings=True)
    scores, ids = index.search(np.asarray(q, dtype="float32"), len(records))
    results = []
    for sc, i in zip(scores[0], ids[0]):
        if i == -1 or sc < min_score:
            continue
        r = records[i]
        if allowed and r["source"] not in allowed:
            continue
        results.append({**r, "score": float(sc)})
        if len(results) >= k:
            break
    return results


def build_retrieval_query(question: str, history: list[dict]) -> str:
    """ถ้าคำถามสั้น (เช่น 'แล้วต้องรดน้ำบ่อยไหม') ให้ต่อกับคำถามก่อนหน้าเพื่อค้นหาให้ตรงบริบท"""
    if len(question) < 25:
        prev = [m["content"] for m in history if m["role"] == "user"]
        if prev:
            return f"{prev[-1]} {question}"
    return question


# ----------------------------- LLM -----------------------------
def get_api_key() -> str | None:
    key = (
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )
    if key:
        return key

    try:
        key = st.secrets.get("GEMINI_API_KEY", "") or st.secrets.get("GOOGLE_API_KEY", "")
    except FileNotFoundError:
        return None
    return key.strip() or None


def generate_answer(
    question: str,
    contexts: list[dict],
    history: list[dict],
    language: str,
) -> str:
    key = get_api_key()
    if not key:
        if language == "en":
            return "⚠️ GEMINI_API_KEY is not configured. Add it to Streamlit Secrets or the environment."
        return "⚠️ ยังไม่ได้ตั้งค่า GEMINI_API_KEY ใน Streamlit Secrets หรือ environment"
    if not contexts:
        if language == "en":
            return "No information found in the documents."
        return "ไม่พบข้อมูลในเอกสาร"

    ctx_text = "\n\n".join(
        f"[{i}] (file: {c['source']} | topic: {c['title']})\n{c['text']}"
        for i, c in enumerate(contexts, 1)
    )

    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history[-HISTORY_TURNS:]
    )
    user_msg = (
        f"Reference documents:\n{ctx_text}\n\n"
        f"Recent conversation:\n{history_text or '(none)'}\n\n"
        f"Question: {question}\n\nAnswer using the system rules."
    )
    language_instruction = (
        'Answer in English only. If the answer is not in the references, say exactly "No information found in the documents."'
        if language == "en"
        else 'ตอบเป็นภาษาไทยเท่านั้น หากไม่มีคำตอบในเอกสาร ให้พูดว่า "ไม่พบข้อมูลในเอกสาร"'
    )
    try:
        client = genai.Client(api_key=key)
        response = client.interactions.create(
            model=LLM_MODEL,
            input=user_msg,
            system_instruction=f"{SYSTEM_PROMPT}\n{language_instruction}",
            generation_config={"temperature": 0.1, "max_output_tokens": 2500},
            store=False,
        )
        text = (response.output_text or "").strip()
        if text:
            return text
        if language == "en":
            return "⚠️ Gemini returned an empty response. Please try again."
        return "⚠️ Gemini ไม่ได้ส่งคำตอบกลับมา กรุณาลองใหม่อีกครั้ง"
    except Exception as e:
        if language == "en":
            return f"⚠️ Gemini request failed: {e}"
        return f"⚠️ เรียก Gemini ไม่สำเร็จ: {e}"


# ----------------------------- UI -----------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;500;600&family=Prompt:wght@500;600;700&display=swap');
:root{--leaf:#173B2D;--leaf2:#2F7D4F;--sprout:#9CCB3B;--mist:#F5F9F2;--side:#EAF1E4;
      --line:#D5E3CB;--ink:#1D2B22;--muted:#5E6F63;--amber:#B7791F;}
.stApp, .stApp button, .stApp textarea, .stApp input{font-family:'IBM Plex Sans Thai','Noto Sans Thai',Tahoma,sans-serif;}
.stApp{color:var(--ink);}
[data-testid="stHeader"]{background:transparent;}
footer{visibility:hidden;}
.block-container{max-width:820px;padding-top:1.2rem;padding-bottom:6rem;}

/* ---------- แถบข้าง (แบบ Claude: สว่าง เรียบ) ---------- */
[data-testid="stSidebar"]{background:var(--side);border-right:1px solid var(--line);}
[data-testid="stSidebar"] .stButton>button{background:transparent;border:none;border-radius:10px;color:var(--ink);
  justify-content:flex-start;text-align:left;padding:.45rem .7rem;font-size:.92rem;line-height:1.4;min-height:2.2rem;}
[data-testid="stSidebar"] .stButton>button:hover{background:rgba(23,59,45,.09);}
[data-testid="stSidebar"] .stButton>button:focus-visible{outline:3px solid var(--sprout);outline-offset:1px;}
[class*="st-key-act_"] button{background:#D3E4C4!important;font-weight:600!important;}
.st-key-newchat button{background:var(--leaf)!important;color:#fff!important;justify-content:center!important;font-weight:600!important;padding:.6rem .8rem!important;margin:.3rem 0 .4rem;}
.st-key-newchat button p{color:#fff!important;}
.st-key-newchat button:hover{background:var(--leaf2)!important;}
[data-testid="stSidebar"] [data-testid="stExpander"]{border:none!important;background:transparent!important;}
[data-testid="stSidebar"] [data-testid="stExpander"] summary{padding:.35rem .4rem;border-radius:10px;font-weight:600;}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover{background:rgba(23,59,45,.07);}
.brand{display:flex;align-items:center;gap:.65rem;margin:.1rem 0 .6rem;}
.brand span{font-size:1.6rem;}
.brand b{font-family:'Prompt',sans-serif;font-size:1.1rem;color:var(--leaf);display:block;line-height:1.2;}
.brand small{color:var(--muted);font-size:.78rem;}
.side-h{font-family:'Prompt',sans-serif;font-weight:600;font-size:.82rem;margin:1rem 0 .25rem;color:var(--muted);padding-left:.4rem;}
.side-note{font-size:.74rem;color:var(--muted);line-height:1.6;margin-top:.8rem;padding:0 .4rem;}

/* ---------- หัวแชต ---------- */
.crumb{font-family:'Prompt',sans-serif;font-weight:500;color:var(--leaf);font-size:1rem;padding-top:.35rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.crumb span{color:var(--muted);margin:0 .25rem;}
.scope{display:inline-block;margin:.1rem 0 .8rem;color:var(--muted);font-size:.82rem;}
.scope b{color:var(--leaf2);font-weight:600;}

/* ---------- Hero ซองเมล็ดพันธุ์ (หน้าแชตใหม่) ---------- */
.hero{background:#fff;border:1px solid var(--line);border-radius:6px 6px 22px 22px;overflow:hidden;margin:.4rem 0 1.2rem;}
.hero-crimp{height:12px;background:repeating-linear-gradient(90deg,var(--leaf) 0 9px,var(--leaf2) 9px 18px);}
.hero-body{display:flex;gap:1.1rem;align-items:center;padding:1.5rem 1.6rem .9rem;}
.hero-badge{flex:0 0 auto;width:64px;height:64px;border-radius:50%;background:var(--sprout);display:flex;align-items:center;justify-content:center;font-size:2rem;}
.hero-title{font-family:'Prompt',sans-serif;font-weight:700;font-size:1.9rem;line-height:1.2;color:var(--leaf);margin:0;}
.hero-sub{margin:.35rem 0 0;color:var(--muted);font-size:1rem;line-height:1.6;max-width:56ch;}
.hero-meta{display:flex;flex-wrap:wrap;gap:.5rem;padding:0 1.6rem 1.3rem;}
.pill{background:var(--mist);border:1px solid var(--line);color:var(--leaf);border-radius:999px;padding:.2rem .8rem;font-size:.85rem;font-weight:500;}
.start-h{font-family:'Prompt',sans-serif;font-weight:600;font-size:1.02rem;color:var(--leaf);margin:1rem 0 .6rem;}
[data-testid="stMain"] .stButton>button{background:#fff;border:1px solid var(--line);border-radius:14px;color:var(--ink);
  justify-content:flex-start;text-align:left;padding:.85rem 1rem;min-height:4.2rem;width:100%;line-height:1.5;}
[data-testid="stMain"] .stButton>button:hover{border-color:var(--leaf2);color:var(--leaf);background:#FBFDF9;}
[data-testid="stMain"] .stButton>button:focus-visible{outline:3px solid var(--sprout);outline-offset:2px;}

/* ---------- ข้อความแชต: ผู้ช่วยเป็นข้อความเรียบ ผู้ใช้เป็นฟองอ่อน ---------- */
[data-testid="stChatMessage"]{background:transparent;border:none;padding:.5rem 0;margin-bottom:.4rem;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){background:#E6F0DA;border-radius:18px;padding:.8rem 1.1rem;}
[data-testid="stChatMessage"] p{line-height:1.75;font-size:1.02rem;}
.chip{display:inline-block;border-radius:999px;padding:.12rem .75rem;font-size:.8rem;font-weight:600;margin-bottom:.5rem;}
.chip-ok{background:#E2F2E6;color:#1F6B3B;border:1px solid #BFE0C8;}
.chip-warn{background:#FFF3DC;color:var(--amber);border:1px solid #F1D79E;}

/* ---------- การ์ดแหล่งอ้างอิง ---------- */
.src{background:var(--mist);border:1px solid var(--line);border-left:5px solid var(--leaf2);border-radius:4px 14px 14px 4px;padding:.8rem 1rem;margin:.55rem 0;}
.src-head{display:flex;align-items:center;gap:.75rem;}
.src-n{flex:0 0 auto;width:28px;height:28px;border-radius:50%;background:var(--leaf);color:#fff;font-weight:600;display:flex;align-items:center;justify-content:center;font-size:.9rem;}
.src-t{flex:1 1 auto;min-width:0;display:flex;flex-direction:column;}
.src-t b{color:var(--leaf);font-size:.98rem;}
.src-t small{color:var(--muted);font-size:.78rem;}
.src-score{flex:0 0 auto;text-align:right;}
.src-score small{color:var(--muted);font-size:.75rem;}
.bar{width:84px;height:6px;background:#E1ECD8;border-radius:99px;overflow:hidden;margin-bottom:2px;}
.bar i{display:block;height:100%;background:var(--sprout);border-radius:99px;}
.src p{margin:.6rem 0 0;color:#33443A;font-size:.92rem;line-height:1.7;}

/* ---------- ช่องพิมพ์ ---------- */
[data-testid="stChatInput"]>div{border-radius:26px;border:1.5px solid var(--line);background:#fff;}
[data-testid="stChatInput"]>div:focus-within{border-color:var(--leaf2);box-shadow:0 0 0 3px rgba(156,203,59,.35);}
@media (max-width:640px){.hero-title{font-size:1.5rem}.hero-body{flex-direction:column;align-items:flex-start}}
"""

st.set_page_config(page_title="Home Vegetable Garden | สวนครัวในบ้าน", page_icon="🌱", layout="centered",
                   initial_sidebar_state="expanded")
ui_language = st.sidebar.selectbox(
    "ภาษาหน้าเว็บ / Interface language",
    options=["ไทย", "English"],
    index=0,
    key="ui_language",
)


def tr(thai: str, english: str) -> str:
    return english if ui_language == "English" else thai


S = st.session_state
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

with st.spinner(tr("กำลังโหลดคู่มือและสร้างดัชนี (ครั้งแรกอาจใช้เวลาสักครู่)...",
                   "Loading the guides and building the search index (this may take a moment the first time)...")):
    _, _, _records = build_index()
DOC_TITLES: dict[str, str] = {}
for _r in _records:
    DOC_TITLES.setdefault(_r["source"], _r["title"])
DOC_TITLES_EN = {
    "01_intro_and_planning.txt": "Introduction and Planning",
    "02_soil_and_potting_mix.txt": "Soil and Potting Mix",
    "03_watering.txt": "Watering",
    "04_fertilizer.txt": "Fertilizer",
    "05_pests_and_diseases.txt": "Pests and Diseases",
    "06_herbs_basil_chili.txt": "Herbs: Basil and Chili",
    "07_leafy_vegetables.txt": "Leafy Vegetables",
    "08_sprouts_microgreens.txt": "Sprouts and Microgreens",
    "09_compost_eco_enzyme.txt": "Compost and Eco-Enzyme",
    "10_seasons_troubleshooting.txt": "Seasons and Troubleshooting",
    "11_balcony_hydroponics_seeds.txt": "Balcony Gardening, Hydroponics, and Seeds",
}
N_DOCS, N_CHUNKS = len(DOC_TITLES), len(_records)


def display_doc_title(source: str) -> str:
    if ui_language == "English":
        return DOC_TITLES_EN.get(source, DOC_TITLES.get(source, source))
    return DOC_TITLES.get(source, source)


def display_project_name(name: str) -> str:
    return tr("ทั่วไป", "General") if name == "ทั่วไป" else name


def display_chat_title(title: str) -> str:
    return tr("แชตใหม่", "New chat") if title == "แชตใหม่" else title


# ----------------------------- สถานะ: โปรเจกต์ / แชต -----------------------------
def esc(x) -> str:
    return html.escape(str(x))


def default_pid() -> str:
    """โปรเจกต์ปลายทางเริ่มต้น: 'general' ถ้ามี ไม่งั้นใช้อันแรก"""
    if "general" in S.projects:
        return "general"
    return next(iter(S.projects))


def new_chat(project_id: str | None = None):
    project_id = project_id if project_id in S.projects else default_pid()
    cur = S.chats.get(S.get("active"))
    if cur and not cur["messages"] and cur["project"] == project_id:
        return  # แชตว่างอยู่แล้ว ไม่ต้องสร้างซ้ำ
    cid = uuid.uuid4().hex[:8]
    S.chats[cid] = {"title": "แชตใหม่", "project": project_id, "messages": [], "ts": time.time()}
    S.active = cid


def set_active(cid: str):
    S.active = cid


def delete_chat(cid: str):
    S.chats.pop(cid, None)
    if S.active == cid:
        if S.chats:
            S.active = max(S.chats, key=lambda c: S.chats[c]["ts"])
        else:
            new_chat(default_pid())


def rename_chat(cid: str):
    name = S.get(f"rn_{cid}", "").strip()
    if name:
        S.chats[cid]["title"] = name[:60]


def move_chat(cid: str):
    S.chats[cid]["project"] = S[f"mv_{cid}"]


def create_project():
    name = S.get("np_name", "").strip() or "โปรเจกต์ใหม่"
    pid = uuid.uuid4().hex[:8]
    S.projects[pid] = {"name": name[:40], "docs": list(S.get("np_docs", []))}
    S.np_name, S.np_docs = "", []
    new_chat(pid)


def save_project(pid: str):
    name = S.get(f"pn_{pid}", "").strip()
    if name:
        S.projects[pid]["name"] = name[:40]
    S.projects[pid]["docs"] = list(S.get(f"pd_{pid}", []))


def delete_project(pid: str):
    if len(S.projects) <= 1 or pid not in S.projects:
        return  # ต้องเหลืออย่างน้อย 1 โปรเจกต์
    S.projects.pop(pid)
    target = default_pid()
    for c in S.chats.values():
        if c["project"] == pid:
            c["project"] = target


def set_pending(q: str):
    S.pending = q


# ----- เก็บประวัติข้ามการรีเฟรช: ผูกกับรหัสใน URL (?u=...) และเก็บเป็นไฟล์ฝั่งเซิร์ฟเวอร์ -----
def _get_user_id() -> str:
    u = st.query_params.get("u", "")
    if not re.fullmatch(r"[a-f0-9]{12}", str(u)):
        u = uuid.uuid4().hex[:12]
        st.query_params["u"] = u
    return u


USER_ID = _get_user_id()
HIST_FILE = os.path.join(tempfile.gettempdir(), "rag_garden_history", f"{USER_ID}.json")


def save_history():
    try:
        os.makedirs(os.path.dirname(HIST_FILE), exist_ok=True)
        with open(HIST_FILE, "w", encoding="utf-8") as f:
            json.dump({"projects": S.projects, "chats": S.chats, "active": S.active}, f, ensure_ascii=False)
    except OSError as e:
        st.warning(tr(f"บันทึกประวัติแชตไม่สำเร็จ: {e}",
                      f"Could not save chat history: {e}"))


def load_history():
    try:
        with open(HIST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not isinstance(data.get("projects"), dict) or not isinstance(data.get("chats"), dict):
            raise ValueError("รูปแบบไฟล์ประวัติไม่ถูกต้อง")
        S.projects, S.chats, S.active = data["projects"], data["chats"], data.get("active")
    except FileNotFoundError:
        return
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        st.warning(tr(f"โหลดประวัติแชตไม่สำเร็จ: {e}",
                      f"Could not load chat history: {e}"))


def heal_state():
    """ซ่อมข้อมูลที่ผิดปกติ เช่น แชตชี้ไปโปรเจกต์ที่ถูกลบ ไม่ให้แอปล้ม"""
    if not S.projects:
        S.projects["general"] = {"name": "ทั่วไป", "docs": []}
    for c in S.chats.values():
        if c.get("project") not in S.projects:
            c["project"] = default_pid()
        c.setdefault("messages", [])
        c.setdefault("title", "แชตใหม่")
        c.setdefault("ts", time.time())
    if S.get("active") not in S.chats:
        S.active = max(S.chats, key=lambda c: S.chats[c]["ts"]) if S.chats else None
    if S.active is None:
        new_chat(default_pid())


if "projects" not in S:
    S.projects = {"general": {"name": "ทั่วไป", "docs": []}}
    S.chats = {}
    S.active = None
    load_history()
heal_state()


def import_history(up):
    data = json.load(up)
    projects, chats = data["projects"], data["chats"]
    if not projects:
        projects["general"] = {"name": "ทั่วไป", "docs": []}
    S.projects, S.chats = projects, chats
    S.active = max(chats, key=lambda c: chats[c]["ts"]) if chats else None
    heal_state()


def status_chip(answer: str) -> str:
    if answer.startswith("⚠️"):
        return ""
    no_info_markers = (
        "ไม่พบข้อมูลในเอกสาร",
        "no information found",
        "the documents do not contain",
        "not found in the documents",
    )
    if any(marker in answer[:120].lower() for marker in no_info_markers):
        return f'<span class="chip chip-warn">{tr("ไม่พบข้อมูลในเอกสาร", "No information found in the documents")}</span>'
    return f'<span class="chip chip-ok">{tr("ตอบจากเอกสารอ้างอิง", "Answered from reference documents")}</span>'


def render_sources(sources: list[dict]):
    with st.expander(tr(f"📚 แหล่งอ้างอิง {len(sources)} รายการ", f"📚 {len(sources)} references")):
        if ui_language == "English":
            st.caption("Reference excerpts are shown in their original language (Thai).")
        if not sources:
            st.write(tr("ไม่มีเอกสารที่ผ่านเกณฑ์ที่ตั้งไว้", "No documents passed the selected threshold."))
        for i, s_ in enumerate(sources, 1):
            pct = max(0, min(100, int((s_["score"] - 0.70) / 0.25 * 100)))
            st.markdown(
                "".join([
                    '<div class="src"><div class="src-head">',
                    f'<span class="src-n">{i}</span>',
                    f'<div class="src-t"><b>{esc(display_doc_title(s_["source"]))}</b>',
                    f'<small>{esc(s_["source"])} {tr("ส่วนที่", "chunk")} {s_["chunk_id"]}</small></div>',
                    f'<div class="src-score"><div class="bar"><i style="width:{pct}%"></i></div>',
                    f'<small>{tr("ความคล้าย", "Similarity")} {s_["score"]:.2f}</small></div></div>',
                    f'<p>{esc(s_["text"])}</p></div>',
                ]),
                unsafe_allow_html=True,
            )


def short(t: str, n: int = 27) -> str:
    return t if len(t) <= n else t[: n - 1] + "…"


def chat_button(cid: str, prefix: str):
    active = cid == S.active
    st.button(
        short(display_chat_title(S.chats[cid]["title"])),
        key=f"{'act' if active else 'chat'}_{prefix}_{cid}",
        on_click=set_active, args=(cid,), width="stretch",
    )


# ----------------------------- Sidebar -----------------------------
active_chat = S.chats[S.active]
with st.sidebar:
    st.markdown(
        f'<div class="brand"><span>🌿</span><div><b>{tr("สวนครัวในบ้าน", "Home Vegetable Garden")}</b>'
        f'<small>{tr("คู่มือปลูกผักแบบถาม-ตอบ", "Your gardening guide")}</small></div></div>',
        unsafe_allow_html=True,
    )
    st.button(tr("＋  แชตใหม่", "＋  New chat"), key="newchat", on_click=new_chat,
              args=(active_chat["project"],), width="stretch")

    recent = sorted([c for c in S.chats if S.chats[c]["messages"]],
                    key=lambda c: -S.chats[c]["ts"])[:5]
    if recent:
        st.markdown(f'<div class="side-h">{tr("ล่าสุด", "Recent")}</div>', unsafe_allow_html=True)
        for cid in recent:
            chat_button(cid, "rc")

    st.markdown(f'<div class="side-h">{tr("โปรเจกต์", "Projects")}</div>', unsafe_allow_html=True)
    for pid, p in list(S.projects.items()):
        in_proj = sorted([c for c in S.chats if S.chats[c]["project"] == pid],
                         key=lambda c: -S.chats[c]["ts"])
        with st.expander(f"📁 {display_project_name(p['name'])} ({len(in_proj)} {tr('แชต', 'chats')})", expanded=(active_chat["project"] == pid)):
            st.button(tr("＋ แชตใหม่ในโปรเจกต์นี้", "＋ New chat in this project"), key=f"np_{pid}", on_click=new_chat,
                      args=(pid,), width="stretch")
            for cid in in_proj:
                chat_button(cid, "pj")
            with st.popover(tr("⚙️ ตั้งค่าโปรเจกต์", "⚙️ Project settings"), width="stretch"):
                st.text_input(tr("ชื่อโปรเจกต์", "Project name"), value=p["name"], key=f"pn_{pid}")
                st.multiselect(
                    tr("จำกัดเอกสารที่ค้นหา (ว่าง = ทุกเรื่อง)", "Limit search to documents (empty = all)"),
                    options=list(DOC_TITLES),
                    default=[d for d in p["docs"] if d in DOC_TITLES],
                    format_func=display_doc_title,
                    key=f"pd_{pid}",
                )
                st.button(tr("บันทึก", "Save"), key=f"ps_{pid}", on_click=save_project, args=(pid,))
                if len(S.projects) > 1:
                    st.button(tr("🗑️ ลบโปรเจกต์ (แชตจะย้ายไปโปรเจกต์อื่น)", "🗑️ Delete project (chats will be moved)"), key=f"pdel_{pid}",
                              on_click=delete_project, args=(pid,))
                else:
                    st.caption(tr("ต้องมีอย่างน้อย 1 โปรเจกต์ สร้างโปรเจกต์ใหม่ก่อนจึงจะลบอันนี้ได้",
                                  "At least one project is required. Create another before deleting this one."))

    with st.expander(tr("＋ โปรเจกต์ใหม่", "＋ New project")):
        st.text_input(tr("ชื่อโปรเจกต์", "Project name"), key="np_name",
                      placeholder=tr("เช่น ปลูกกะเพราและพริก", "e.g. Basil and chili"))
        st.multiselect(
            tr("จำกัดเอกสารที่ค้นหา (ว่าง = ทุกเรื่อง)", "Limit search to documents (empty = all)"),
            options=list(DOC_TITLES),
            format_func=display_doc_title,
            key="np_docs",
        )
        st.button(tr("สร้างโปรเจกต์", "Create project"), key="np_create", on_click=create_project)

    st.markdown(f'<div class="side-h">{tr("ลองถามดู", "Try asking")}</div>', unsafe_allow_html=True)
    with st.expander(tr("ตัวอย่างคำถาม", "Example questions")):
        questions = EXAMPLE_QUESTIONS if ui_language == "ไทย" else EXAMPLE_QUESTIONS_EN
        for q in questions:
            st.button(q, key=f"ex_{q}", on_click=set_pending, args=(q,), width="stretch")

    st.markdown(f'<div class="side-h">{tr("ตั้งค่า", "Settings")}</div>', unsafe_allow_html=True)
    with st.expander(tr("ปรับการค้นหาเอกสาร", "Search settings")):
        top_k = st.slider(tr("จำนวนแหล่งอ้างอิง (Top-K)", "Number of references (Top-K)"), 1, 8, 4)
        min_score = st.slider(tr("ค่าความคล้ายต่ำสุด (0 = ไม่กรอง)", "Minimum similarity (0 = no filter)"), 0.0, 0.95, 0.0, 0.05)
    with st.expander(tr("สำรอง / นำเข้าประวัติ", "Export / import chat history")):
        st.caption(tr(
            "ประวัติผูกกับลิงก์ที่มี ?u=... ต่อท้าย บุ๊กมาร์กลิงก์นี้ไว้เพื่อกลับมาดูแชตเดิม (ถ้าแอปถูกรีสตาร์ต ประวัติอาจหาย จึงควรกดสำรองเป็นไฟล์ไว้ด้วย)",
            "Chat history is linked to the ?u=... URL. Bookmark the link to return to it. Server restarts may clear history, so download a backup.",
        ))
        st.download_button(
            tr("⬇️ สำรองประวัติ (.json)", "⬇️ Download history (.json)"),
            data=json.dumps({"projects": S.projects, "chats": S.chats}, ensure_ascii=False, indent=1),
            file_name="chat_history.json", mime="application/json", width="stretch",
        )
        up = st.file_uploader(tr("นำเข้าประวัติ (.json)", "Import history (.json)"), type="json", key="imp")
        if up is not None:
            fid = getattr(up, "file_id", f"{up.name}-{up.size}")
            if S.get("imp_id") != fid:
                try:
                    import_history(up)
                    S.imp_id = fid
                    st.rerun()
                except Exception:  # noqa: BLE001
                    st.error(tr("ไฟล์ไม่ถูกต้อง นำเข้าไม่สำเร็จ",
                                "Invalid file. Could not import chat history."))
    st.markdown(
        f'<div class="side-note">Embedding: {esc(EMBED_MODEL)}<br>AI: Google Gemini ({esc(LLM_MODEL)})<br>'
        f'{tr("ภาษาคำตอบปรับตามภาษาคำถาม", "Answers follow the language of your question")}</div>',
        unsafe_allow_html=True,
    )

# ----------------------------- หน้าหลัก -----------------------------
heal_state()
save_history()
cid = S.active
chat = S.chats[cid]
proj = S.projects[chat["project"]]
scope = [d for d in proj["docs"] if d in DOC_TITLES]

top_l, top_r = st.columns([6, 1.3], vertical_alignment="center")
with top_l:
    st.markdown(f'<div class="crumb">📁 {esc(display_project_name(proj["name"]))}<span>/</span>{esc(display_chat_title(chat["title"]))}</div>',
                unsafe_allow_html=True)
with top_r:
    with st.popover(tr("⋯ จัดการ", "⋯ Manage"), width="stretch"):
        st.text_input(tr("ชื่อแชต", "Chat name"), value=chat["title"], key=f"rn_{cid}")
        st.button(tr("บันทึกชื่อ", "Save name"), key=f"rnb_{cid}", on_click=rename_chat, args=(cid,))
        st.selectbox(tr("ย้ายไปโปรเจกต์", "Move to project"), options=list(S.projects), index=list(S.projects).index(chat["project"]),
                     format_func=lambda x: display_project_name(S.projects[x]["name"]), key=f"mv_{cid}",
                     on_change=move_chat, args=(cid,))
        st.button(tr("🗑️ ลบแชตนี้", "🗑️ Delete this chat"), key=f"del_{cid}", on_click=delete_chat, args=(cid,))

if scope:
    names = ", ".join(display_doc_title(d) for d in scope[:2]) + (
        tr(f" และอีก {len(scope) - 2} เรื่อง", f" and {len(scope) - 2} more")
        if len(scope) > 2 else ""
    )
    st.markdown(
        f'<div class="scope">{tr("ค้นหาจาก", "Searching")} <b>{len(scope)} '
        f'{tr("เรื่อง", "topics")}</b> {tr("ที่โปรเจกต์เลือกไว้", "selected for this project")}: {esc(names)}</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="scope">{tr("ค้นหาจากคู่มือ", "Searching all")} '
        f'<b>{tr("ทุกเรื่อง", "topics")}</b> ({N_DOCS} {tr("เอกสาร", "documents")})</div>',
        unsafe_allow_html=True,
    )

STARTERS = [
    ("☀️", "ปลูกผักในกระถางต้องการแดดวันละกี่ชั่วโมง?"),
    ("🪴", "สูตรดินผสมสำหรับปลูกผักในกระถางคืออะไร?"),
    ("💧", "ควรรดน้ำผักวันละกี่ครั้ง ช่วงไหนดีที่สุด?"),
    ("🐛", "กำจัดเพลี้ยอ่อนแบบไม่ใช้สารเคมีทำอย่างไร?"),
    ("🌿", "ใบกะเพราเหลืองเกิดจากอะไร?"),
    ("♻️", "ทำปุ๋ยหมักจากเศษอาหารอย่างไร?"),
]
STARTERS_EN = [
    ("☀️", "How many hours of sunlight do vegetables in pots need?"),
    ("🪴", "What is a good potting mix for vegetables?"),
    ("💧", "How often and when should I water vegetables?"),
    ("🐛", "How can I control aphids without chemicals?"),
    ("🌿", "Why are my holy basil leaves turning yellow?"),
    ("♻️", "How can I make compost from kitchen scraps?"),
]
if not chat["messages"] and "pending" not in S:
    starters = STARTERS if ui_language == "ไทย" else STARTERS_EN
    st.markdown(
        "".join([
            '<div class="hero"><div class="hero-crimp"></div><div class="hero-body">',
            '<div class="hero-badge">🌱</div><div>',
            f'<h1 class="hero-title">{tr("ผู้ช่วยปลูกผักสวนครัว", "Home Vegetable Gardening Assistant")}</h1>',
            f'<p class="hero-sub">{tr("ถามเรื่องแสง ดิน น้ำ ปุ๋ย และโรคแมลง ได้คำตอบจากคู่มือพร้อมแหล่งอ้างอิง ถ้าในคู่มือไม่มีข้อมูล ระบบจะบอกตรง ๆ", "Ask about sunlight, soil, watering, fertilizer, pests, and diseases. Answers cite the guides, and the assistant will say when information is unavailable.")}</p></div></div>',
            f'<div class="hero-meta"><span class="pill">{tr("คู่มือ", "Guides")} {N_DOCS} {tr("เรื่อง", "topics")}</span>',
            f'<span class="pill">{N_CHUNKS} {tr("ส่วนความรู้", "knowledge chunks")}</span>',
            f'<span class="pill">{tr("ภาษาไทยและอังกฤษ", "Thai and English")}</span></div></div>',
        ]),
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="start-h">{tr("เริ่มจากคำถามยอดนิยม", "Start with a popular question")}</div>',
                unsafe_allow_html=True)
    cols = st.columns(2)
    for idx, (icon, q) in enumerate(starters):
        with cols[idx % 2]:
            st.button(f"{icon}  {q}", key=f"st_{idx}", on_click=set_pending, args=(q,))

for m in chat["messages"]:
    avatar = "🧑‍🌾" if m["role"] == "user" else "🌱"
    with st.chat_message(m["role"], avatar=avatar):
        if m["role"] == "assistant":
            st.markdown(status_chip(m["content"]), unsafe_allow_html=True)
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_sources(m.get("sources", []))

question = st.chat_input(tr(
    "ถามเรื่องผัก เช่น ใบกะเพราเหลืองเกิดจากอะไร?",
    "Ask about gardening, e.g. Why are my basil leaves turning yellow?",
))
if not question and "pending" in S:
    question = S.pop("pending")

if question:
    history = list(chat["messages"])
    chat["messages"].append({"role": "user", "content": question})
    chat["ts"] = time.time()
    if chat["title"] == "แชตใหม่":
        chat["title"] = short(question, 40)
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🌱"):
        with st.spinner(tr("กำลังเปิดคู่มือและเรียบเรียงคำตอบ...",
                           "Searching the guides and preparing an answer...")):
            language = detect_language(question)
            ctxs = retrieve(build_retrieval_query(question, history), top_k, min_score, scope or None)
            answer = generate_answer(question, ctxs, history, language)
        st.markdown(status_chip(answer), unsafe_allow_html=True)
        st.markdown(answer)
        render_sources(ctxs)

    chat["messages"].append({"role": "assistant", "content": answer, "sources": ctxs})
    save_history()
    st.rerun()  # รีเฟรชแถบข้างให้ชื่อแชตและรายการล่าสุดอัปเดตทันที
