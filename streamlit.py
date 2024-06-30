import os
import google.generativeai as genai
import pandas as pd
import _notes.streamlit as st
import requests
from io import BytesIO
from dotenv import load_dotenv
import numpy as np
import re
from tqdm import tqdm

load_dotenv()
API_KEY = os.getenv("API_KEY")

genai.configure(api_key=API_KEY)

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
    safety_settings=safety_settings,
    generation_config=generation_config,
    #     system_instruction = """
    # Greeting and Introduction:
    # - "Hello, welcome to Jorakay. My name is "น้องจระเข้". How can I assist you today?"
    # Initial Inquiry:
    # - "What type of product information are you looking for?"
    # - "Or, what kind of construction project do you need product recommendations for?"
    # Analyzing Customer Requirements:
    # - "How large is your project?"
    # - "Or, what specific type of materials are you looking for?"
    # Product Recommendation:
    # - "I recommend [Product Name] because it's suitable for your project's durability and value."
    # - "The [Product Name] features [specify features] and comes with a [specify warranty terms]."
    # Answering Customer Questions:
    # - "What kind of weather conditions and installation methods does this product suit?"
    # - "I understand your concern about [mention customer concern]. Our product addresses [mention benefits or problem-solving features]."
    # Closing the Sale:
    # - "Currently, we have special promotions for orders placed this month."
    # - "Let me confirm your order details. You require [Product Name], quantity [quantity], priced at [price]. Is that correct?"
    # Follow-up and Customer Service:
    # - "Thank you for choosing our service."
    # - "Feel free to contact us anytime if you have further questions or issues."
    # """
    system_instruction="""
Operation Method:
    You will receive product information in Row-LIST format (users don't need to know the source).
    Respond to user questions about products politely and directly.

Guidelines for Response:
    Politeness: Use "คะ" or "ค่ะ" when communicating with users.
    Relevance: Focus only on details relevant to the user's question.
    Insufficient data: If there's not enough or the table is empty, show understanding and inform the user that there's no information.

Special cases:
    Explain differences between product models.
    Provide examples for clarity.

Data extraction:
    If other columns have no data, extract key details from "Display_Name," emphasizing specifics like size and weight over general names.
    Calculations: Basic calculations can assist in answering questions.

Data consistency:
    "Display_Name" column may contain complete data (e.g., size, color, brand, model).
    Use data from columns like "Color," "Type," "Model," "Series," "Sizing" if available.
    If these columns lack data, use "Display_Name."

Data inconsistency:
    "Color," "Sizing," "Series," "Material" may appear in the "Display_Name" column.
    Use "Display_Name" data as supplementary information.

Response Format:
    Provide useful information for smooth shopping experience.
    Format text orderly and clearly (use line breaks, bullet points, or other formats).
    Clearly and concisely answer questions.
    Avoid listing long product names; summarize and select relevant data.
    Optionally decline to answer if information is insufficient.

Example:
User: "ขอรายละเอียด ซีเมนต์กันซึมอเนกประสงค์ หน่อยครับ"
Assistant: "จระเข้ อีซี่ ซีเมนต์กันซึมอเนกประสงค์ ซีเมนต์ทากันซึมแบบส่วนผสมเดียว ใช้งานง่ายเพียงผสมน้ำ ทนต่อสภาวะอากาศได้ดี ปล่อยเปลือยได้
ทนรังสี UV มีความคงทน ไม่หลุดล่อน มีแรงยึดเกาะที่ดี ไม่มีสารพิษ เหมาะสำหรับงานกันรั่วซึม ดาดฟ้า ระเบียง ห้องน้ำ บ่อเลี้ยงปลา 
สามารถใช้เป็นปูนกันซึมทาก่อนปูกระเบื้อง หรือ ทาทับกระเบื้องเดิมที่รั่วซึมก่อนปูกระเบื้องใหม่ทับได้ (โดยไม่ต้องรื้อกระเบื้อง) 
และยังสามารถทาบนวัสดุอื่นๆ เช่น โลหะ ไม้ เรซิ่น โฟม เพื่อเปลี่ยนพื้นผิวเป็นคอนกรีตได้อีกด้วยค่ะ"

User: "มีกี่ขนาดครับ"
Assistant: "มีขนาดกระป๋องละ 1 กก. ค่ะ"

User: "ไม่แน่ใจว่ามีของพร้อมจำหน่ายไหมครับ"
Assistant: "มีค่ะ เด๋วทางหนูจะขอรับออเดอร์ไปตรวจสอบก่อนนะ" """,
)


def clear_history():
    st.session_state["messages"] = [
        {"role": "model", "content": "สวัสดีค่ะ ต้องการสอบถามอะไรบ้างคะ"}
    ]
    st.experimental_rerun()

def finetune():
    # Load data
    df = pd.read_excel("data/product_jorakay_all.xlsx")

    # Select and clean relevant columns
    columns = [
        "P_name",
        "P_name_eng",
        "P_detail",
        "P_property",
        "P_guide",
        "P_contain_shown",
        "P_contain_unit",
        "P_color_name",
        "P_netprice",
        "P_sku_code",
        "P_img_link",
    ]
    df = df[columns].drop_duplicates().replace(np.nan, "-", regex=True)


    # Clean text data
    def clean_text(text):
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


    df = df.applymap(lambda x: clean_text(str(x)))

    # Create SKU details
    df["sku_details"] = df.apply(
        lambda x: f"Color: {x.P_color_name} Sizing: {x.P_contain_shown} Unit: {x.P_contain_unit} "
        f"Price: {x.P_netprice} บาท SKUs Code: {x.P_sku_code} Image Link: {x.P_img_link}",
        axis=1,
    )

    # Aggregate product details
    product_details = []
    for _, row in tqdm(
        df[["P_name", "P_name_eng", "P_detail", "P_property", "P_guide"]]
        .drop_duplicates()
        .iterrows()
    ):
        data = df[
            (df.P_name == row.P_name)
            & (df.P_name_eng == row.P_name_eng)
            & (df.P_detail == row.P_detail)
            & (df.P_property == row.P_property)
            & (df.P_guide == row.P_guide)
        ]
        sku_details = "\n".join(data.sku_details.unique())
        desc = (
            f"Product: {row.P_name}\n"
            f"Product Name: {row.P_name_eng}\n"
            f"Description: {row.P_detail}\n"
            f"Feature: {row.P_property}\n"
            f"How to use: {row.P_guide}\n"
            f"Product variant:\n{sku_details}"
        )
        product_details.append(desc)

    # Create a DataFrame with only the products column
    output_df = pd.DataFrame({"products": product_details})

    # Save the cleaned product details
    output_df.to_csv("data/clean_product_details.txt", index=False)
    print("Save complete")

with st.sidebar:
    if st.button("Fine Turning"):
        finetune()
        
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

file_path = "data/clean_product_details.txt"
try:
    df = pd.read_csv(file_path)
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
