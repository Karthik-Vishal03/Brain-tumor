## Brain-tumor
This project uses three different models and k fold cross validation

# Dataset collection: - 
There are around 7031 MRI images obtained from Kaggle and other sources. There are 5716 training images and 1315 testing images. These images are classified into 4 different classes namely – no tumor, glioma tumor, meningioma tumor and pituitary tumor. The images are 256x256 pixel in size. They are in grayscale images to make the data augmentation an easier process.

# Data Preprocessing: - 
The pre-processing is image augmentation which is done by a keras module named ‘ImageDataGenerator’. This module specializes in image augmentation while you’re still training your model. So, the images are resized and passed into the dataframe so this way the model is robust and saves up overhead memory. Also, there are changes made in the brightness and the contrast of the images. There is a use of train-test split, k cross validation and mini batch training processes used to split the data.

# Model selection and Training: -
Since the dataset has images, had to go with 2 different Deep learning models – one pre-trained, VGG16 and the other one is Convolutional Neural Network (CNN).
1.	CNN: - CNN is a network architecture that leans by itself. It is very helpful in learning from the data, analysing the common patterns from the data and make the classification easier, that’s why this is mostly used in deep learning models.
2.	VGG16: - This is also a CNN, but it is pre-trained so makes building of weights during the training process easier. It consists of 16 layers. This is very much useful for image classification tasks as it can learn complex features.
Hyperparameter tunning: - This can be done by changing the learning rate of the VGG16 model. By changing the value of the number of folds. By changing the activation functions of the CNN model.

# Model Testing & Evaluation: 
The model is applied to the testing data and evaluated using metrics such as accuracy, F1-score, recall, precision, etc.

# GUI Development: 
The Web Application is created using Streamlit by python and deploying the best model saved in a h5 (hierarchical data format) file (.h5) which is employed by keras to save models in this format. The test data can be predicted using the Application whether the person has brain tumor or not and which type of tumor it is.
