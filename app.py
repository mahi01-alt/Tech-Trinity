import streamlit as st
import google.generativeai as genai
from PIL import Image
import numpy as np
import cv2

# ----------------------------------
# 🔑 PASTE YOUR GOOGLE API KEY HERE
# ----------------------------------
GOOGLE_API_KEY = "AIzaSyBviJAG-Z3WH-X3ktq-jgjKFq4pQdoGm54"

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# ----------------------------------
# Streamlit Page Config
# ----------------------------------
st.set_page_config(page_title="StyleAI - Fashion Advisor", layout="wide")

st.title("👗 StyleAI - AI Powered Fashion Recommendation System")
st.write("Upload your photo and get personalized fashion recommendations!")

# ----------------------------------
# Skin Tone Detection Function
# ----------------------------------
def detect_skin_tone(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (300, 300))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img_rgb.shape
    center_region = img_rgb[h//4:int(h/1.5), w//4:int(w/1.5)]

    avg_color = np.mean(center_region.reshape(-1, 3), axis=0)
    r, g, b = avg_color
    brightness = (r + g + b) / 3

    if brightness > 200:
        tone = "Fair"
    elif brightness > 150:
        tone = "Medium"
    elif brightness > 100:
        tone = "Olive"
    else:
        tone = "Deep"

    return tone, (int(r), int(g), int(b))


# ----------------------------------
# AI Recommendation Function
# ----------------------------------
def generate_recommendation(skin_tone, gender):
    prompt = f"""
    You are a professional fashion stylist.

    User Details:
    - Skin Tone: {skin_tone}
    - Gender: {gender}

    Provide:
    1. Best color palette (primary, secondary, accent)
    2. Outfit recommendations (Formal, Casual, Party)
    3. Accessories suggestions
    4. Hairstyle suggestions
    5. Explanation of why these suit the skin tone
    6. Suggest Indian shopping brands (Amazon, Myntra, Zara)

    Give structured and clean response.
    """

    response = model.generate_content(prompt)
    return response.text


# ----------------------------------
# UI Layout
# ----------------------------------
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Your Photo", type=["jpg", "png", "jpeg"])
    gender = st.selectbox("Select Gender", ["Male", "Female"])

with col2:
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze & Get Recommendations"):
            with st.spinner("Analyzing your style..."):

                skin_tone, rgb = detect_skin_tone(image)

                st.subheader("🎨 Skin Tone Analysis")
                st.write(f"Detected Skin Tone: **{skin_tone}**")
                st.write(f"RGB Values: {rgb}")

                recommendation = generate_recommendation(skin_tone, gender)

                st.subheader("✨ AI Fashion Recommendations")
                st.write(recommendation)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + Google Gemini AI")
