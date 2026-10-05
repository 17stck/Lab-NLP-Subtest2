"""
🌱 ผู้ช่วยปลูกผักสวนครัวในบ้าน — RAG Chatbot
Document Loading & Chunking -> Embedding + FAISS -> Prompt Engineering -> Groq LLM -> Streamlit chat
"""
import glob
import html
import json
import os
import re
import time
import uuid

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pythainlp.util import normalize
from sentence_transformers import SentenceTransformer

# ----------------------------- การตั้งค่า -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBED_MODEL = "intfloat/multilingual-e5-small"  # โมเดลเล็ก รองรับไทย/อังกฤษ
LLM_MODEL = "openai/gpt-oss-120b"                # โมเดลหลักบน Groq
FALLBACK_MODELS = ["openai/gpt-oss-20b", "llama-3.1-8b-instant"]  # สลับอัตโนมัติถ้าโมเดลหลักใช้ไม่ได้
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

SYSTEM_PROMPT = """คุณคือ "ผู้ช่วยปลูกผักสวนครัว" ที่ตอบคำถามโดยอิงจากเอกสารความรู้ที่ให้มาเท่านั้น

กฎที่ต้องปฏิบัติอย่างเคร่งครัด:
1. ตอบจาก "เอกสารอ้างอิง" ที่ให้มาเท่านั้น ห้ามใช้ความรู้นอกเอกสาร ห้ามเดาหรือแต่งข้อมูลเพิ่ม
2. ทุกประโยคที่เป็นข้อเท็จจริงต้องมีเลขอ้างอิงในรูปแบบ [1], [2] ต่อท้าย ตรงกับหมายเลขของเอกสารที่ใช้
3. หากเอกสารไม่มีข้อมูลที่ตอบคำถามได้ ให้ตอบว่า "ไม่พบข้อมูลในเอกสาร" พร้อมอธิบายสั้น ๆ ว่าเอกสารครอบคลุมเรื่องใดบ้าง และห้ามอ้างอิงเลขเอกสาร
4. หากคำถามต่อเนื่องจากบทสนทนาก่อนหน้า ให้ใช้ประวัติการสนทนาเพื่อเข้าใจบริบท แต่ข้อเท็จจริงต้องมาจากเอกสารอ้างอิงเท่านั้น
5. ตอบเป็นภาษาเดียวกับผู้ใช้ กระชับ เป็นขั้นตอน เข้าใจง่าย
"""


# ----------------------------- โหลด + ทำความสะอาด + แบ่ง chunk -----------------------------
def clean_text(text: str) -> str:
    text = normalize(text)                       # จัดระเบียบสระ/วรรณยุกต์ซ้ำของภาษาไทย
    text = text.replace("\u200b", "").replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_long(unit: str, size: int) -> list[str]:
    """ตัดย่อหน้ายาวเป็นประโยค (ใช้ pythainlp) แล้วตัดแข็งถ้ายังยาวเกิน"""
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
            # overlap: ยกหน่วยสุดท้ายไปซ้อน ถ้าไม่ยาวเกินไป
            last = current.split(" ")[-1] if " " in current else ""
            carry = last if 0 < len(last) <= CHUNK_OVERLAP else ""
            current = (carry + " " + u).strip()
        else:
            current = (current + " " + u).strip()
    if current:
        chunks.append(current)
    return title, chunks


@st.cache_resource(show_spinner="กำลังโหลดเอกสารและสร้างดัชนี (ครั้งแรกอาจใช้เวลาสักครู่)...")
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
        raise RuntimeError("ไม่พบไฟล์ .txt ในโฟลเดอร์ data/")

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
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY")


def generate_answer(question: str, contexts: list[dict], history: list[dict]) -> str:
    key = get_api_key()
    if not key:
        return "⚠️ ยังไม่ได้ตั้งค่า GROQ_API_KEY ใน Secrets ของ Streamlit"

    if contexts:
        ctx_text = "\n\n".join(
            f"[{i}] (ไฟล์: {c['source']} | หัวข้อ: {c['title']})\n{c['text']}"
            for i, c in enumerate(contexts, 1)
        )
    else:
        ctx_text = "(ไม่พบเอกสารที่เกี่ยวข้อง)"

    user_msg = f"เอกสารอ้างอิง:\n{ctx_text}\n\nคำถาม: {question}\n\nตอบตามกฎที่กำหนด:"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history[-HISTORY_TURNS:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_msg})

    client = Groq(api_key=key)
    last_err = None
    for model_name in [LLM_MODEL] + FALLBACK_MODELS:
        try:
            kwargs = dict(model=model_name, messages=messages, temperature=0.1, max_tokens=2500)
            if model_name.startswith("openai/gpt-oss"):
                kwargs["extra_body"] = {"reasoning_effort": "low"}  # ลดเวลา/โทเคนที่ใช้คิด
            resp = client.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    return f"⚠️ เรียก LLM ไม่สำเร็จ: {last_err}"


# ----------------------------- UI -----------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;500;600&family=Prompt:wght@500;600;700&display=swap');
:root{--leaf:#173B2D;--leaf2:#2F7D4F;--sprout:#9CCB3B;--mist:#F5F9F2;--side:#EAF1E4;
      --line:#D5E3CB;--ink:#1D2B22;--muted:#5E6F63;--amber:#B7791F;}
.stApp, .stApp button, .stApp textarea, .stApp input{font-family:'IBM Plex Sans Thai',sans-serif;}
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

st.set_page_config(page_title="ผู้ช่วยปลูกผักสวนครัว", page_icon="🌱", layout="centered",
                   initial_sidebar_state="expanded")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

_, _, _records = build_index()
DOC_TITLES: dict[str, str] = {}
for _r in _records:
    DOC_TITLES.setdefault(_r["source"], _r["title"])
N_DOCS, N_CHUNKS = len(DOC_TITLES), len(_records)
S = st.session_state

# ----------------------------- สถานะ: โปรเจกต์ / แชต -----------------------------
def esc(x) -> str:
    return html.escape(str(x))


def new_chat(project_id: str = "general"):
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
            new_chat("general")


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
    if pid == "general":
        return
    for c in S.chats.values():
        if c["project"] == pid:
            c["project"] = "general"
    S.projects.pop(pid, None)


def set_pending(q: str):
    S.pending = q


if "projects" not in S:
    S.projects = {"general": {"name": "ทั่วไป", "docs": []}}
    S.chats = {}
    S.active = None
    new_chat("general")


def import_history(up):
    data = json.load(up)
    projects, chats = data["projects"], data["chats"]
    projects.setdefault("general", {"name": "ทั่วไป", "docs": []})
    for c in chats.values():
        if c.get("project") not in projects:
            c["project"] = "general"
    S.projects, S.chats = projects, chats
    S.active = max(chats, key=lambda c: chats[c]["ts"]) if chats else None
    if S.active is None:
        new_chat("general")


def status_chip(answer: str) -> str:
    if answer.startswith("⚠️"):
        return ""
    if "ไม่พบข้อมูลในเอกสาร" in answer[:60]:
        return '<span class="chip chip-warn">ไม่พบข้อมูลในเอกสาร</span>'
    return '<span class="chip chip-ok">ตอบจากเอกสารอ้างอิง</span>'


def render_sources(sources: list[dict]):
    with st.expander(f"📚 แหล่งอ้างอิง {len(sources)} รายการ"):
        if not sources:
            st.write("ไม่มีเอกสารที่ผ่านเกณฑ์ที่ตั้งไว้")
        for i, s_ in enumerate(sources, 1):
            pct = max(0, min(100, int((s_["score"] - 0.70) / 0.25 * 100)))
            st.markdown(
                "".join([
                    '<div class="src"><div class="src-head">',
                    f'<span class="src-n">{i}</span>',
                    f'<div class="src-t"><b>{esc(s_["title"])}</b>',
                    f'<small>{esc(s_["source"])} ส่วนที่ {s_["chunk_id"]}</small></div>',
                    f'<div class="src-score"><div class="bar"><i style="width:{pct}%"></i></div>',
                    f'<small>ความคล้าย {s_["score"]:.2f}</small></div></div>',
                    f'<p>{esc(s_["text"])}</p></div>',
                ]),
                unsafe_allow_html=True,
            )


def short(t: str, n: int = 27) -> str:
    return t if len(t) <= n else t[: n - 1] + "…"


def chat_button(cid: str, prefix: str):
    active = cid == S.active
    st.button(
        short(S.chats[cid]["title"]),
        key=f"{'act' if active else 'chat'}_{prefix}_{cid}",
        on_click=set_active, args=(cid,), use_container_width=True,
    )


# ----------------------------- Sidebar -----------------------------
active_chat = S.chats[S.active]
with st.sidebar:
    st.markdown(
        '<div class="brand"><span>🌿</span><div><b>สวนครัวในบ้าน</b>'
        '<small>คู่มือปลูกผักแบบถาม-ตอบ</small></div></div>',
        unsafe_allow_html=True,
    )
    st.button("＋  แชตใหม่", key="newchat", on_click=new_chat,
              args=(active_chat["project"],), use_container_width=True)

    recent = sorted([c for c in S.chats if S.chats[c]["messages"]],
                    key=lambda c: -S.chats[c]["ts"])[:5]
    if recent:
        st.markdown('<div class="side-h">ล่าสุด</div>', unsafe_allow_html=True)
        for cid in recent:
            chat_button(cid, "rc")

    st.markdown('<div class="side-h">โปรเจกต์</div>', unsafe_allow_html=True)
    for pid, p in list(S.projects.items()):
        in_proj = sorted([c for c in S.chats if S.chats[c]["project"] == pid],
                         key=lambda c: -S.chats[c]["ts"])
        with st.expander(f"📁 {p['name']} ({len(in_proj)})", expanded=(active_chat["project"] == pid)):
            st.button("＋ แชตใหม่ในโปรเจกต์นี้", key=f"np_{pid}", on_click=new_chat,
                      args=(pid,), use_container_width=True)
            for cid in in_proj:
                chat_button(cid, "pj")
            with st.popover("⚙️ ตั้งค่าโปรเจกต์", use_container_width=True):
                st.text_input("ชื่อโปรเจกต์", value=p["name"], key=f"pn_{pid}")
                st.multiselect("จำกัดเอกสารที่ค้นหา (ว่าง = ทุกเรื่อง)", options=list(DOC_TITLES),
                               default=[d for d in p["docs"] if d in DOC_TITLES],
                               format_func=lambda x: DOC_TITLES[x], key=f"pd_{pid}")
                st.button("บันทึก", key=f"ps_{pid}", on_click=save_project, args=(pid,))
                if pid != "general":
                    st.button("ลบโปรเจกต์ (แชตจะย้ายไป 'ทั่วไป')", key=f"pdel_{pid}",
                              on_click=delete_project, args=(pid,))

    with st.expander("＋ โปรเจกต์ใหม่"):
        st.text_input("ชื่อโปรเจกต์", key="np_name", placeholder="เช่น ปลูกกะเพราและพริก")
        st.multiselect("จำกัดเอกสารที่ค้นหา (ว่าง = ทุกเรื่อง)", options=list(DOC_TITLES),
                       format_func=lambda x: DOC_TITLES[x], key="np_docs")
        st.button("สร้างโปรเจกต์", key="np_create", on_click=create_project)

    st.markdown('<div class="side-h">ลองถามดู</div>', unsafe_allow_html=True)
    with st.expander("ตัวอย่างคำถาม"):
        for q in EXAMPLE_QUESTIONS:
            st.button(q, key=f"ex_{q}", on_click=set_pending, args=(q,), use_container_width=True)

    st.markdown('<div class="side-h">ตั้งค่า</div>', unsafe_allow_html=True)
    with st.expander("ปรับการค้นหาเอกสาร"):
        top_k = st.slider("จำนวนแหล่งอ้างอิง (Top-K)", 1, 8, 4)
        min_score = st.slider("ค่าความคล้ายต่ำสุด (0 = ไม่กรอง)", 0.0, 0.95, 0.0, 0.05)
    with st.expander("สำรอง / นำเข้าประวัติ"):
        st.caption("ประวัติแชตเก็บในเซสชันนี้เท่านั้น ถ้ารีเฟรชหน้าจะหาย กดสำรองไว้แล้วนำเข้าใหม่ได้")
        st.download_button(
            "⬇️ สำรองประวัติ (.json)",
            data=json.dumps({"projects": S.projects, "chats": S.chats}, ensure_ascii=False, indent=1),
            file_name="chat_history.json", mime="application/json", use_container_width=True,
        )
        up = st.file_uploader("นำเข้าประวัติ (.json)", type="json", key="imp")
        if up is not None:
            fid = getattr(up, "file_id", f"{up.name}-{up.size}")
            if S.get("imp_id") != fid:
                try:
                    import_history(up)
                    S.imp_id = fid
                    st.rerun()
                except Exception:  # noqa: BLE001
                    st.error("ไฟล์ไม่ถูกต้อง นำเข้าไม่สำเร็จ")
    st.markdown(
        f'<div class="side-note">Embedding: {esc(EMBED_MODEL)}<br>LLM: {esc(LLM_MODEL)} (Groq)</div>',
        unsafe_allow_html=True,
    )

# ----------------------------- หน้าหลัก -----------------------------
cid = S.active
chat = S.chats[cid]
proj = S.projects[chat["project"]]
scope = [d for d in proj["docs"] if d in DOC_TITLES]

top_l, top_r = st.columns([6, 1.3], vertical_alignment="center")
with top_l:
    st.markdown(f'<div class="crumb">📁 {esc(proj["name"])}<span>/</span>{esc(chat["title"])}</div>',
                unsafe_allow_html=True)
with top_r:
    with st.popover("⋯ จัดการ", use_container_width=True):
        st.text_input("ชื่อแชต", value=chat["title"], key=f"rn_{cid}")
        st.button("บันทึกชื่อ", key=f"rnb_{cid}", on_click=rename_chat, args=(cid,))
        st.selectbox("ย้ายไปโปรเจกต์", options=list(S.projects), index=list(S.projects).index(chat["project"]),
                     format_func=lambda x: S.projects[x]["name"], key=f"mv_{cid}",
                     on_change=move_chat, args=(cid,))
        st.button("🗑️ ลบแชตนี้", key=f"del_{cid}", on_click=delete_chat, args=(cid,))

if scope:
    names = ", ".join(DOC_TITLES[d] for d in scope[:2]) + (f" และอีก {len(scope) - 2} เรื่อง" if len(scope) > 2 else "")
    st.markdown(f'<div class="scope">ค้นหาจาก <b>{len(scope)} เรื่อง</b> ที่โปรเจกต์เลือกไว้: {esc(names)}</div>',
                unsafe_allow_html=True)
else:
    st.markdown(f'<div class="scope">ค้นหาจากคู่มือ <b>ทุกเรื่อง</b> ({N_DOCS} เอกสาร)</div>', unsafe_allow_html=True)

STARTERS = [
    ("☀️", "ปลูกผักในกระถางต้องการแดดวันละกี่ชั่วโมง?"),
    ("🪴", "สูตรดินผสมสำหรับปลูกผักในกระถางคืออะไร?"),
    ("💧", "ควรรดน้ำผักวันละกี่ครั้ง ช่วงไหนดีที่สุด?"),
    ("🐛", "กำจัดเพลี้ยอ่อนแบบไม่ใช้สารเคมีทำอย่างไร?"),
    ("🌿", "ใบกะเพราเหลืองเกิดจากอะไร?"),
    ("♻️", "ทำปุ๋ยหมักจากเศษอาหารอย่างไร?"),
]
if not chat["messages"] and "pending" not in S:
    st.markdown(
        "".join([
            '<div class="hero"><div class="hero-crimp"></div><div class="hero-body">',
            '<div class="hero-badge">🌱</div><div>',
            '<h1 class="hero-title">ผู้ช่วยปลูกผักสวนครัว</h1>',
            '<p class="hero-sub">ถามเรื่องแสง ดิน น้ำ ปุ๋ย และโรคแมลง ได้คำตอบจากคู่มือพร้อมแหล่งอ้างอิง '
            'ถ้าในคู่มือไม่มีข้อมูล ระบบจะบอกตรง ๆ</p></div></div>',
            f'<div class="hero-meta"><span class="pill">คู่มือ {N_DOCS} เรื่อง</span>',
            f'<span class="pill">{N_CHUNKS} ส่วนความรู้</span>',
            '<span class="pill">ภาษาไทยและอังกฤษ</span></div></div>',
        ]),
        unsafe_allow_html=True,
    )
    st.markdown('<div class="start-h">เริ่มจากคำถามยอดนิยม</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for idx, (icon, q) in enumerate(STARTERS):
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

question = st.chat_input("ถามเรื่องผัก เช่น ใบกะเพราเหลืองเกิดจากอะไร?")
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
        with st.spinner("กำลังเปิดคู่มือและเรียบเรียงคำตอบ..."):
            ctxs = retrieve(build_retrieval_query(question, history), top_k, min_score, scope or None)
            answer = generate_answer(question, ctxs, history)
        st.markdown(status_chip(answer), unsafe_allow_html=True)
        st.markdown(answer)
        render_sources(ctxs)

    chat["messages"].append({"role": "assistant", "content": answer, "sources": ctxs})
    st.rerun()  # รีเฟรชแถบข้างให้ชื่อแชตและรายการล่าสุดอัปเดตทันที
