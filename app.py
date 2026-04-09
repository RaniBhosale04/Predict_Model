import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- Page Configuration & Animations ---
st.set_page_config(page_title="AI Impact Predictor", page_icon="✨", layout="centered")

# Custom CSS for fade-in and button animations
st.markdown("""
<style>
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .stApp {
        animation: fadeIn 1.2s ease-out;
    }
    .stButton>button {
        transition: all 0.3s ease;
        border-radius: 8px;
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        color: white;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("✨ AI Tool Usage Prediction")
st.write("Enter the user's details below to generate a prediction using the KNN model.")
st.markdown("---")

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Error: 'model.pkl' not found. Please ensure it is in the same directory.")
        return None

model = load_model()

# --- Input UI ---
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=20)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    city = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])

with col2:
    ai_tool = st.selectbox("AI Tool Used", ["ChatGPT", "Gemini", "Claude", "Other"])
    daily_hours = st.slider("Daily Usage (Hours)", 0.0, 24.0, 2.0, 0.5)
    purpose = st.selectbox("Primary Purpose", ["Academic", "Professional", "Entertainment", "Other"])
    impact = st.selectbox("Impact on Grades", ["Positive", "Neutral", "Negative"])

# --- IMPORTANT: Data Mapping ---
# Replace these dictionaries with the EXACT numerical encoding used when training your model!
encode_dict = {
    "gender": {"Male": 0, "Female": 1, "Other": 2},
    "education": {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3},
    "city": {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2},
    "ai_tool": {"ChatGPT": 0, "Gemini": 1, "Claude": 2, "Other": 3},
    "purpose": {"Academic": 0, "Professional": 1, "Entertainment": 2, "Other": 3},
    "impact": {"Positive": 0, "Neutral": 1, "Negative": 2}
}

# --- Prediction Logic ---
if st.button("Generate Prediction 🚀"):
    if model is not None:
        # 1. Format inputs
        input_data = {
            "Age": age,
            "Gender": encode_dict["gender"][gender],
            "Education_Level": encode_dict["education"][education],
            "City": encode_dict["city"][city],
            "AI_Tool_Used": encode_dict["ai_tool"][ai_tool],
            "Daily_Usage_Hours": daily_hours,
            "Purpose": encode_dict["purpose"][purpose],
            "Impact_on_Grades": encode_dict["impact"][impact]
        }
        
        # 2. Convert to DataFrame to retain feature names (required by modern scikit-learn)
        input_df = pd.DataFrame([input_data])
        
        # 3. Predict
        try:
            prediction = model.predict(input_df)[0]
            
            # Show animation & result
            st.balloons()
            st.success(f"### Prediction Result: {prediction}")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
