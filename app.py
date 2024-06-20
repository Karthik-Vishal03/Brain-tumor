import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


st.set_page_config(
    page_title="Brain Tumour Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="auto",
)

st.sidebar.image("tumor.jpeg", caption="Health is Wealth")

# CSS to customize the sidebar, background image, and radio button font size
st.markdown(
    """
    <style>
    /* Adjust the background image */
    .stApp {
        background: url("https://www.drmboyi.co.za/wp-content/uploads/2022/06/Dr-Mboyi-Blog-Templete-Brain-Cancer.jpg");
        background-size: cover;
        background-position: left center; /* Move the background image to the left and center it vertically */
    }

    /* Adjust the font size of radio button labels */
    div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 24px; /* Change the font size to 24px */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the saved model
model = tf.keras.models.load_model("model.h5")

# Preprocessing function
def preprocess_image(image):
    # Resize image to match model input size
    resized_image = image.resize((299, 299))
    # Convert image to numpy array and normalize
    image_array = np.array(resized_image) / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    # Add channel dimension
    image_array = np.expand_dims(image_array, axis=-1)
    return image_array

# Streamlit app
st.write("<h1 style='color:black'>Brain Tumor Classification 🧠</h1>", unsafe_allow_html=True)

# Sidebar options
option = st.sidebar.radio("Navigation", ("Home", "Prediction", "Tracker"))

# Home page
if option == "Home":
    st.write("<h3 style='color:black'>HOMEPAGE</h3>", unsafe_allow_html=True)
    st.write("<p style='color:black'>This is the home page of the Brain Tumour Classifier app.</p>", unsafe_allow_html=True)
    
    # Information about brain tumors
    st.write("<h2 style='color:black'>Brain Tumors</h2>", unsafe_allow_html=True)
    st.write("<p style='color:black'>A brain tumor is a mass or growth of abnormal cells in the brain. Tumors can be either benign (non-cancerous) "
                "or malignant (cancerous). They can originate in the brain itself or spread from other parts of the body.</p>", unsafe_allow_html=True)
    
    # Symptoms and signs
    st.write("<h2 style='color:black'>Symptoms and Signs</h2>", unsafe_allow_html=True)
    st.write("<p style='color:black'>Common symptoms of brain tumors include headaches, seizures, changes in vision, difficulty speaking or "
                "understanding speech, and changes in mood or personality.</p>", unsafe_allow_html=True)
    
    # How to monitor brain health and seek help
    st.write("<h2 style='color:black'>How to Monitor Brain Health and Seek Help</h2>", unsafe_allow_html=True)
    st.write("<p style='color:black'>1. Regular medical check-ups and screenings can help monitor brain health.</p>", unsafe_allow_html=True)
    st.write("<p style='color:black'>2. Pay attention to any unusual symptoms and seek medical advice if you notice persistent changes.</p>", unsafe_allow_html=True)
    st.write("<p style='color:black'>3. Early detection and treatment are crucial for better outcomes.</p>", unsafe_allow_html=True)

    st.write("<h3 style='color:black'>Model used to predict the classification is Convolution Neural Network and we have acchieved an accuracy of <b>98.93%🎯</b> in predicting the tumor.</h3>", unsafe_allow_html=True)



# Prediction page
elif option == "Prediction":
    st.write("<h1 style='color:black'>Brain Tumor Prediction </h1>", unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        preprocessed_image = preprocess_image(image)

       # Perform inference
        prediction = model.predict(preprocessed_image)
        class_index = np.argmax(prediction)
        classes = ["GLIOMA TUMOR 🔴", "MENINGIOMA TUMOR 🔴", "NO TUMOR 🥳", "PITUITARY TUMOR 🔴"]

       # Check if the predicted class is "NO TUMOR"
        if class_index == 2:
           message = "No tumor detected. 🎉"
        else:
           message = "Consult a doctor for further evaluation. 🩺"

       # Highlight prediction result
        st.markdown(
        f"""
        <div style="background-color:#f0f0f0;padding:10px;border-radius:5px">
        <h1 style='color:black'>Prediction Result</h1>
        <h3 style='color:black'><strong>Predicted Class:</strong> {classes[class_index]}</h3>
        <h3 style='color:black'><strong>Confidence:</strong> {prediction[0][class_index] * 100:.2f}%</h3>
        <p style='color:black'>{message}</p>
        </div>
         """,
    unsafe_allow_html=True
)

elif option=='Tracker':

 def generate_sample_data(num_patients=5, num_months=12):
    data = {
        'Patient_ID': [],
        'Month': [],
        'Tumor_Size_cm': []
    }
    for patient_id in range(1, num_patients + 1):
        for month in range(1, num_months + 1):
            data['Patient_ID'].append(patient_id)
            data['Month'].append(month)
            # Generate random tumor size (cm) data
            data['Tumor_Size_cm'].append(np.random.randint(1, 10))
    return pd.DataFrame(data)

 def main():
    st.title('Brain Tumor Progression Tracker')

    # Generate sample data
    df = generate_sample_data()

    # Sidebar - Patient selection
    selected_patient = st.sidebar.selectbox("Select Patient ID", df['Patient_ID'].unique())

    # Filter data for selected patient
    filtered_df = df[df['Patient_ID'] == selected_patient]

    # Line plot for tumor size progression
    st.subheader("Tumor Size Progression")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(filtered_df['Month'], filtered_df['Tumor_Size_cm'], marker='o', linestyle='-')
    ax.set_xlabel('Month')
    ax.set_ylabel('Tumor Size (cm)')
    ax.set_title(f'Tumor Size Progression for Patient {selected_patient}')
    st.pyplot(fig)

    # Data table for tumor size data
    st.subheader("Tumor Size Data")
    st.write(filtered_df)

 if __name__ == "__main__":
    main()




