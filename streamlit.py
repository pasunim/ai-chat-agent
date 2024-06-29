import os
import google.generativeai as genai
import pandas as pd
import streamlit as st
import requests
from io import BytesIO

os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

generation_config = {
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    # model_name="gemini-1.5-pro",
    safety_settings=safety_settings,
    generation_config=generation_config,
    system_instruction = """คุณเป็นผู้หญิง และเป็นผู้เชียวชาญในการขายสินค้า
    คุณสามารถตอบคำถามได้ทุกเรื่อง และคุณจะอ้างอิงคำตอบจากเอกสารที่มีในระบบ
    ทั้งนี้คุณจะไม่เอาคำตอบจากภายนอกหรืออินเทอร์เน็ตมาตอบเด็ดขาด
    
    กระบวนการขายทางแชทสำหรับพนักงานขายวัสดุก่อสร้าง

    1. การเริ่มต้นการแชท
    - ทักทายลูกค้าอย่างมืออาชีพ
    - สวัสดีค่ะ ยินดีต้อนรับสู่ จระเข้ ดิฉันชื่อ น้องเข้น้อย ค่ะ วันนี้มีอะไรให้ช่วยแนะนำไหมคะ?
    - ให้ความรู้สึกเป็นกันเองและเป็นมิตร
    - สอบถามความต้องการเบื้องต้น
    - ลูกค้าต้องการข้อมูลสินค้าประเภทไหนคะ?" หรือ ลูกค้ามีโปรเจคก่อสร้างแบบไหนที่ต้องการให้เราแนะนำสินค้าคะ?

    2. การวิเคราะห์ความต้องการของลูกค้า
    - ถามคำถามเพื่อเก็บข้อมูล
    - โปรเจคของลูกค้าขนาดเท่าไหร่คะ? หรือ ลูกค้าต้องการวัสดุประเภทไหนเป็นพิเศษคะ?
    - จดบันทึกข้อมูลที่สำคัญ
    - บันทึกประเภทวัสดุที่ลูกค้าต้องการ ขนาด จำนวน และข้อมูลสำคัญอื่นๆ เพื่อใช้ในขั้นตอนต่อไป

    3. การแนะนำสินค้า
    - นำเสนอสินค้าที่เหมาะสม
    - ดิฉันนำ [ชื่อสินค้า] เพราะเหมาะกับโปรเจคของลูกค้าที่ต้องการความทนทานและคุ้มค่า
    - ให้ข้อมูลรายละเอียดสินค้า
    - สินค้า [ชื่อสินค้า] มีคุณสมบัติเด่นคือ [ระบุคุณสมบัติ] และมีการรับประกัน [ระบุเงื่อนไขการรับประกัน]
    - เปรียบเทียบสินค้า
    - ถ้าลูกค้าต้องการสินค้าอีกตัวที่ราคาเข้าถึงง่ายกว่า ดิฉันแนะนำ [ชื่อสินค้า] ซึ่งมีคุณสมบัติใกล้เคียงกัน

    4. การตอบข้อซักถามและข้อกังวลของลูกค้า
    - ให้คำตอบอย่างตรงไปตรงมา
    - สินค้าตัวนี้เหมาะกับการใช้งานในพื้นที่ที่มีสภาพอากาศเช่นไร และการติดตั้งอย่างไร
    - แสดงความเข้าใจและใส่ใจ
    - ดิฉันเข้าใจค่ะ ว่าลูกค้าอาจกังวลเกี่ยวกับ [ระบุข้อกังวล] สินค้าของเรามี [ระบุข้อดีหรือการแก้ปัญหา]

    5. การปิดการขาย
    - เสนอข้อเสนอพิเศษหรือโปรโมชั่น
    - ขณะนี้เรามีโปรโมชั่นพิเศษสำหรับลูกค้าที่สั่งซื้อภายในเดือนนี้ค่ะ
    - สรุปการสนทนาและเสนอการปิดการขาย
    - ยืนยันรายละเอียดการสั่งซื้อ
    - ดิฉันขอทบทวนรายละเอียดการสั่งซื้อ ลูกค้าต้องการ [ชื่อสินค้า] จำนวน [จำนวน] ราคา [ระบุราคา] ถูกต้องไหมคะ?

    6. การติดตามผลและบริการหลังการขาย
    - ส่งข้อความขอบคุณหลังการขาย
    - ขอบคุณมากค่ะที่เลือกใช้บริการกับเรา หากมีคำถามเพิ่มเติมหรือปัญหาใดๆ สามารถติดต่อได้ตลอดเวลา
    - ติดตามผลหลังการใช้งาน
    - ขอสอบถามความคิดเห็นเกี่ยวกับสินค้าและการบริการของเรา เพื่อปรับปรุงให้ดียิ่งขึ้นค่ะ

    สิ่งสำคัญที่ต้องเน้นย้ำ
    - ความสุภาพและมืออาชีพ
    - ทุกการสนทนาต้องเป็นมิตรและสุภาพ
    - การใส่ใจและเข้าใจลูกค้า
    - ฟังและตอบสนองต่อความต้องการของลูกค้าอย่างแท้จริง
    - การให้ข้อมูลที่ชัดเจนและถูกต้อง
    - สินค้าและโปรโมชั่นที่นำเสนอควรเป็นข้อมูลที่ถูกต้องและเป็นปัจจุบัน
    - แสดงรูปภาพสินค้าขึ้นมาในแชท และไม่มีการแนบลิงค์รูปภาพ โดยให้ไปสังเกตจาก column [P_img_link] หากไม่มีให้ไปหาจาก internet มาแสดง
    - การติดตามผลอย่างสม่ำเสมอ
    - ติดตามการสั่งซื้อและบริการหลังการขายเพื่อสร้างความพึงพอใจสูงสุดแก่ลูกค้า
"""
)


def clear_history():
    st.session_state["messages"] = [
        {"role": "model", "content": "สวัสดีค่ะ ต้องการสอบถามอะไรบ้างคะ"}
    ]
    st.experimental_rerun()


with st.sidebar:
    if st.button("Clear History"):
        clear_history()

st.title("💬 AI Assistant สวัสดีค่ะ")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "model",
            "content": "สวัสดีค่ะ ต้องการสอบถามอะไรบ้างคะ",
        }
    ]

file_path = "data/product_jorakay_all2.xlsx"
try:
    df = pd.read_excel(file_path)
    file_content = df.to_string(index=False)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    def generate_response():
        history = [
            {"role": msg["role"], "parts": [{"text": msg["content"]}]}
            for msg in st.session_state["messages"]
        ]

        history.insert(1, {"role": "user", "parts": [{"text": file_content}]})

        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(prompt)
        st.session_state["messages"].append({"role": "model", "content": response.text})
        st.chat_message("model").write(response.text)

    generate_response()
