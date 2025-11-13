Plant Disease Detection Using Convolutional Neural Networks (CNN)
Overview

This project presents an Artificial Intelligence-based system for the automatic detection of plant diseases from leaf images using deep learning.
The main goal is to assist farmers and researchers in early disease identification, preventing large-scale crop losses and improving agricultural productivity.

Dataset

The model is trained on the PlantVillage dataset from Kaggle, which contains 38 classes of healthy and diseased plant leaves.
Images are preprocessed, normalized, and augmented to improve model performance and generalization.

Model Architecture

Framework: TensorFlow / Keras

Model Type: Convolutional Neural Network (CNN)

Layers: Convolutional → ReLU → Pooling → Fully Connected → Softmax

Output: 38-class probability vector (one for each plant disease)

Loss Function: Categorical Crossentropy

Optimizer: Adam

After training, the model is saved as model.h5 for later inference.

How to Run the Project
Step 1: Install Dependencies

Make sure you have Python 3.8 or higher installed. Then run:

pip install tensorflow streamlit pillow numpy

Step 2: Train the Model (Optional)

If you want to retrain the model:

Open and run the notebook Plant_Disease_Prediction_Using_CNN.ipynb in Jupyter Notebook or Google Colab.

The trained model will be saved as model.h5 in your working directory.

Step 3: Run the Streamlit Application

Open the add2.py file.

Inside this file, specify the path to your trained model:

model_path = "path/to/your/model.h5"


This variable is defined directly in add2.py and used to load the trained CNN model.

Run the Streamlit app:

streamlit run add2.py


Upload a leaf image to the Streamlit interface and view the predicted disease class and model confidence score.

Output

The Streamlit application displays:

The uploaded leaf image

The predicted disease name or healthy status

The model’s confidence score

Results

The CNN model achieved high accuracy in classifying plant diseases.

It effectively generalizes to unseen leaf images.
