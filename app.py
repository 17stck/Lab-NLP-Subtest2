"""
🌱 ผู้ช่วยปลูกผักสวนครัวในบ้าน — RAG Chatbot
Document Loading & Chunking -> Embedding + FAISS -> Prompt Engineering -> Groq LLM -> Streamlit chat
"""
import glob
import os
import re

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
def retrieve(query: str, k: int, min_score: float):
    model, index, records = build_index()
    q = model.encode([f"query: {query}"], normalize_embeddings=True)
    scores, ids = index.search(np.asarray(q, dtype="float32"), k)
    results = []
    for s, i in zip(scores[0], ids[0]):
        if i == -1 or s < min_score:
            continue
        r = dict(records[i])
        r["score"] = float(s)
        results.append(r)
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
st.set_page_config(page_title="ผู้ช่วยปลูกผักสวนครัว", page_icon="🌱", layout="centered")
st.title("🌱 ผู้ช่วยปลูกผักสวนครัวในบ้าน")
st.caption("ถามเรื่องการปลูกผักในบ้านได้เลย — ตอบจากเอกสารความรู้เท่านั้น พร้อมแสดงแหล่งอ้างอิงทุกครั้ง")

with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    top_k = st.slider("จำนวนเอกสารที่ดึงมาใช้ (Top-K)", 1, 8, 4)
    min_score = st.slider("ค่าความคล้ายต่ำสุด (0 = ไม่กรอง)", 0.0, 0.95, 0.0, 0.05)
    if st.button("🗑️ ล้างการสนทนา", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.subheader("💡 ตัวอย่างคำถาม")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, key=f"ex_{q}", use_container_width=True):
            st.session_state.pending = q
    st.divider()
    st.caption(f"Embedding: {EMBED_MODEL}\n\nLLM: {LLM_MODEL} (Groq)")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources: list[dict]):
    with st.expander(f"📚 เอกสารอ้างอิงที่ใช้ ({len(sources)} รายการ)"):
        if not sources:
            st.write("ไม่พบเอกสารที่เกี่ยวข้อง")
        for i, s in enumerate(sources, 1):
            st.markdown(f"**[{i}] {s['title']}**  \n`{s['source']}` · chunk {s['chunk_id']} · คะแนนความคล้าย {s['score']:.3f}")
            st.write(s["text"])
            st.divider()


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_sources(m.get("sources", []))

question = st.chat_input("พิมพ์คำถามเกี่ยวกับการปลูกผัก...")
if not question and "pending" in st.session_state:
    question = st.session_state.pop("pending")

if question:
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาและเรียบเรียงคำตอบ..."):
            ctxs = retrieve(build_retrieval_query(question, history), top_k, min_score)
            answer = generate_answer(question, ctxs, history)
        st.markdown(answer)
        render_sources(ctxs)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": ctxs})
