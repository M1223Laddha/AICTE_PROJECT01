import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title='Disease Prediction System', layout='wide', page_icon='🩺')

# Load trained models using correct paths
diabetes_model = pickle.load(open("models/diabetes_model_rf.sav", "rb"))
heart_model = pickle.load(open("models/heart_model002_rf.sav", "rb"))
parkinsons_model = pickle.load(open("models/parkinsons001_model_rf.sav", "rb"))

# Option menu for navigation
selected = option_menu(
    menu_title='Disease Prediction System',
    options=['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinson’s Prediction'],
    icons=['activity', 'heart', 'person'],
    menu_icon='hospital-fill',
    default_index=0,
    orientation='horizontal'
)

# ---------------------- Diabetes Prediction ----------------------
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        SkinThickness = st.text_input('Skin Thickness Value')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    with col2:
        Glucose = st.text_input('Glucose Level')
        Insulin = st.text_input('Insulin Level')
        Age = st.text_input('Age of the Person')
    with col3:
        BloodPressure = st.text_input('Blood Pressure Value')
        BMI = st.text_input('BMI Value')

    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        user_input = [float(x) if x.strip() else 0 for x in user_input]  # Convert empty values to 0
        diab_prediction = diabetes_model.predict([user_input])
        diab_diagnosis = '✅ The Person is NOT Diabetic' if diab_prediction[0] == 0 else '⚠️ The Person is Diabetic'

    st.success(diab_diagnosis)

# ---------------------- Heart Disease Prediction ----------------------
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', min_value=1, max_value=120, step=1)
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=50, max_value=200, step=1)
        chol = st.number_input('Serum Cholesterol (mg/dL)', min_value=100, max_value=600, step=1)
        thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, step=1)
    with col2:
        sex = st.radio('Sex', [0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
        fbs = st.radio('Fasting Blood Sugar > 120 mg/dL', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        exang = st.radio('Exercise Induced Angina', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=6.0, step=0.1)
    with col3:
        cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
        restecg = st.selectbox('Resting ECG Results', [0, 1, 2])
        slope = st.selectbox('Slope of Peak Exercise ST Segment', [0, 1, 2])
        ca = st.number_input('Number of Major Vessels Colored by Fluoroscopy', min_value=0, max_value=3, step=1)
    thal = st.selectbox('Thalassemia Type', [0, 1, 2, 3])

    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        user_input = np.array([float(x) for x in user_input]).reshape(1, -1)  # Ensure correct shape

        # Get probability prediction
        heart_probabilities = heart_model.predict_proba(user_input)[0]
        threshold = 0.4  # Lowered for better sensitivity

        # Debugging prints (Remove in production)
        print(f"Model Classes: {heart_model.classes_}")
        print(f"Prediction Probabilities: {heart_probabilities}")

        if heart_probabilities[1] >= threshold:
            heart_diagnosis = "⚠️ The Person is at Risk of Heart Disease"
        else:
            heart_diagnosis = "✅ The Person does NOT have Heart Disease"

    st.success(heart_diagnosis)


# ---------------------- Parkinson’s Disease Prediction ----------------------
if selected == 'Parkinson’s Prediction':
    st.title('Parkinson’s Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)
    with col1:
        MDVP_Fo = st.text_input('MDVP:Fo (Hz)')
        MDVP_Jitter = st.text_input('MDVP:Jitter (%)')
        MDVP_Shimmer = st.text_input('MDVP:Shimmer')
    with col2:
        MDVP_Fhi = st.text_input('MDVP:Fhi (Hz)')
        MDVP_Jitter_Abs = st.text_input('MDVP:Jitter (Abs)')
        Shimmer_dB = st.text_input('Shimmer dB')
    with col3:
        MDVP_Flo = st.text_input('MDVP:Flo (Hz)')
        HNR = st.text_input('Harmonics-to-Noise Ratio (HNR)')
        RPDE = st.text_input('RPDE')

    parkinsons_diagnosis = ''
    if st.button('Parkinson’s Test Result'):
        user_input = [MDVP_Fo, MDVP_Jitter, MDVP_Shimmer, MDVP_Fhi, MDVP_Jitter_Abs, Shimmer_dB, MDVP_Flo, HNR, RPDE]
        user_input = [float(x) if x.strip() else 0 for x in user_input]  # Convert empty values to 0

        # Ensure all 22 features are included
        user_input_extended = np.append(user_input, [0] * (22 - len(user_input)))

        # Get probability prediction
        probability = parkinsons_model.predict_proba([user_input_extended])[0][1]
        threshold = 0.75  

        if probability >= threshold:
            parkinsons_diagnosis = "⚠️ The Person has Parkinson’s"
        else:
            parkinsons_diagnosis = "✅ The Person does NOT have Parkinson’s"

    st.success(parkinsons_diagnosis)
