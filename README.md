# ผู้ช่วยปลูกผักสวนครัว

เว็บแอปถาม-ตอบเรื่องปลูกผัก โดยค้นข้อมูลจากไฟล์ใน `data/` และใช้ Google Gemini API จาก Google AI Studio เรียบเรียงคำตอบ
รองรับหน้าจอภาษาไทยและ English (เลือกภาษาได้จากแถบด้านข้าง) และตอบกลับเป็นภาษาที่ใช้ถาม
คู่มืออ้างอิงต้นฉบับเป็นภาษาไทย แม้ถามเป็นภาษาอังกฤษคำตอบจะเป็นภาษาอังกฤษ

Home gardening Q&A app with Thai/English interface and answers. Responses follow the
language of the question; source guides are currently written in Thai.

## เริ่มใช้งานในเครื่อง

ติดตั้งแพ็กเกจ:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

ตั้งค่า Google AI Studio API key ใน PowerShell ก่อนเปิดแอป:

```powershell
$env:GEMINI_API_KEY = "ใส่ API key ของคุณ"
.\.venv\Scripts\streamlit.exe run app.py
```

หรือเพิ่ม `GEMINI_API_KEY = "..."` ใน `.streamlit/secrets.toml` โดยห้าม commit ไฟล์ secrets
สร้าง API key ได้จาก [Google AI Studio](https://aistudio.google.com/app/apikey)

## Deploy ให้เข้าจากอินเทอร์เน็ต

1. Push โค้ดขึ้น GitHub โดยให้มี `app.py`, `requirements.txt` และโฟลเดอร์ `data/`
2. สร้างแอปใหม่ที่ [Streamlit Community Cloud](https://share.streamlit.io/) โดยเลือก repository, branch และ `app.py`
3. ในหน้า App settings → Secrets เพิ่ม:

   ```toml
   GEMINI_API_KEY = "ใส่ API key จาก Google AI Studio"
   ```

4. กด Deploy แล้วใช้ URL `https://<ชื่อแอป>.streamlit.app` ที่ระบบสร้างให้

ห้ามใส่ API key ใน source code หรือ commit `.streamlit/secrets.toml` ขึ้น GitHub
การ Deploy แบบสาธารณะทำให้ผู้ที่มีลิงก์เรียกใช้ API quota ของเจ้าของ key ได้
ครั้งแรกแอปจะดาวน์โหลด embedding model จาก Hugging Face และสร้างดัชนีจากเอกสารใน `data/`
