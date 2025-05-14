# 🧠 Brain Tumor Detection

This project deals with the **detection and classification of different types of brain tumors** using MRI images. It employs deep learning models (CNN and VGG16) and uses **Streamlit** to create an interactive web interface for predictions.

---

## 📂 Dataset

- Total Images: **7031 MRI scans**
- Training Set: **5716 images**
- Testing Set: **1315 images**
- Image Size: **256x256 pixels**
- Image Type: **Grayscale**
- Classes:
  - **No Tumor**
  - **Glioma Tumor**
  - **Meningioma Tumor**
  - **Pituitary Tumor**

Dataset sources include **Kaggle** and other medical imaging repositories.

---

## 🧹 Data Preprocessing

- **Image Augmentation** is done using Keras' `ImageDataGenerator`, allowing real-time augmentation during training.
- Techniques applied:
  - Resizing
  - Brightness and contrast adjustment
  - Real-time augmentation
- **Train-Test Split**, **Mini-batch training**, and **K-Fold Cross Validation** are used for robust training and evaluation.

---

## 🧠 Model Selection & Training

This project uses two deep learning architectures:

### 1. 🧬 Convolutional Neural Network (CNN)
- Custom-built architecture from scratch.
- Learns features and patterns directly from MRI data.
- Performs classification based on learned features.

### 2. 🧠 VGG16 (Pre-trained)
- A 16-layer deep Convolutional Neural Network.
- Pre-trained on ImageNet.
- Fine-tuned for brain tumor classification.
- Capable of capturing complex image features.

### 🔧 Hyperparameter Tuning
- Adjusting learning rate
- Varying number of folds in K-Fold validation
- Changing activation functions

---

## 📊 Model Evaluation

The trained models are tested on unseen data using standard performance metrics:

- ✅ Accuracy
- 🔁 F1-Score
- 🔍 Precision
- 📈 Recall

---

## 🖥️ Web Application (GUI)

An interactive GUI was developed using **Streamlit**:

- Accepts MRI image input
- Loads the best-trained model (saved in `.h5` format)
- Predicts the tumor type
- Provides immediate visual feedback to the user

---

## 🛠️ Technologies Used

- 🐍 Python
- 🧠 TensorFlow / Keras
- 🖼️ OpenCV
- 📦 NumPy, Pandas
- 📊 Scikit-learn
- 🌐 Streamlit

---

## 🚀 How to Run

### 1. Clone the Repository

git clone <your-repo-url>
cd brain-tumor-detection

shell
Copy
Edit

### 2. Install Dependencies

pip install -r requirements.txt

shell
Copy
Edit

### 3. Run Streamlit App

streamlit run app.py

yaml
Copy
Edit

---

## 🏁 Results

The project demonstrates strong accuracy in detecting and classifying brain tumors, offering both educational value and real-world potential in medical image analysis.

---

## 🧾 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- Kaggle for dataset access
- Keras & TensorFlow for model building
- Streamlit for GUI development

---

## 📬 Contact

For any queries, feel free to reach out or open an issue.
