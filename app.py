import streamlit as st
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.models import load_model

st.markdown("""
    <style>
    .stApp {
        background-image:  linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)) ,url("https://www.researchtrials.org/wp-content/uploads/2021/03/iStock-1128931450-scaled.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
    }

    h1 {
        color: #FFD700;  /* Gold */
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .glow-text {
        font-size: 50px;
        color: #ffffff;
        text-align: center;
        text-shadow: 0 0 10px #00cfff, 0 0 20px #00cfff, 0 0 30px #00cfff;
        font-weight: bold;
    }
    </style>
    <div class="glow-text">💓 Heart Attack Risk Predictor</div>
    """,
    unsafe_allow_html=True
)

# Load model and scaler
@st.cache_resource
def load_keras_model():
    return load_model('HAP Model.keras')

model = load_keras_model()
scaler = pickle.load(open('HAPM_StandardScaler.pkl', 'rb'))

####################

st.markdown("""
    <style>
    /* Sidebar custom style */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 50, 0.8);  /* Dark blue-ish tone */
        color: white;
    }

    [data-testid="stSidebar"] .css-1v3fvcr {
        color: white;
    }

    /* Optional: make sidebar title/headings colored */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #00cfff;  /* Light cyan */
    }

    /* Optional: control scrollbar style inside sidebar */
    ::-webkit-scrollbar-thumb {
        background: #00cfff;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar.expander("📁 Project Intro"):
    st.markdown("- **This is a Heart Attack Risk Prediction web app using an Artificial Neural Network (ANN)." \
    "It takes medical input features and predicts the likelihood of a heart attack.**")
 

with st.sidebar.expander("👨‍💻 Developer's Intro"):
    st.markdown("- **Hi, I'm Rayyan Ahmed**")
    st.markdown("- **IBM Certifed Advanced LLM FineTuner**")
    st.markdown("- **Google Certified Soft Skill Professional**")
    st.markdown("- **Hugging Face Certified in Fundamentals of Large Language Models (LLMs)**")
    st.markdown("- **Have expertise in EDA, ML, Reinforcement Learning, ANN, CNN, CV, RNN, NLP, LLMs.**")
    st.markdown("[💼Visit Rayyan's LinkedIn Profile](https://www.linkedin.com/in/rayyan-ahmed-504725321/)")

with st.sidebar.expander("🛠️ Tech Stack Used"):
    st.markdown("- **Numpy**")
    st.markdown("- **Pandas**")
    st.markdown("- **Matplotlib**")
    st.markdown("- **Seaborn**")
    st.markdown("- **Scikit Learn**")
    st.markdown("- **TensorFlow, Keras, Pickle**")
    st.markdown("- **Streamlit**")

####################

st.markdown(
    "<h4 style='color: white; font-size: 20px; font-family: Arial; font-weight: bold'>📝 Enter your health information below:</h4>",
    unsafe_allow_html=True
)

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=40)
sex_label = st.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex_label == "Male" else 0
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3], 
                  help="cp type -> 0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic")
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholestoral (chol)", min_value=100, max_value=600, value=240)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], help="1 = True, 0 = False")
restecg = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2], 
                       help="0: Normal, 1: ST-T abnormality, 2: Probable/definite left ventricular hypertrophy")
thalach = st.number_input("Max Heart Rate (thalach)", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1], help="1 = Yes, 0 = No")
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=6.2, value=1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2], help="0: Upsloping, 1: Flat, 2: Downsloping")
ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4], help="Colored by fluoroscopy")
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3], help="1: Fixed Defect, 2: Normal, 3: Reversible Defect")

if st.button("Predict"):
    
    try:
    
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
        
        scaled_features = scaler.transform(features)
        
        prediction = model.predict(scaled_features)[0][0]
        risk = prediction * 100
        
        st.subheader("🩺 Result:")
        if prediction > 0.5:
            st.error(f"⚠️ High Risk of Heart Attack ({risk:.3f}%)")
            st.snow()
            st.markdown(
                """
                <div style="background-color:#ff4d4d; padding:15px; border-radius:10px; color:white; font-size:18px; font-weight:bold;">
                    🚑 <b>Please consult a doctor immediately!</b>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        else:
            st.success(f"✅ Low Risk of Heart Attack ({risk:.3f}%)")
            st.balloons()
            st.markdown(
                """
                <div style="background-color:green; padding:15px; border-radius:10px; color:white; font-size:18px; font-weight:bold;">
                    🧘 <b>You're doing great! Keep maintaining a healthy lifestyle.</b>
                </div>
                """,
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error("❌ Something went wrong during prediction.")
        st.code(str(e))

