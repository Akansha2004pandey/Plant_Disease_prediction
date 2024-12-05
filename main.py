import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.imagenet_utils import preprocess_input

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to 224x224 as expected by the models
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = preprocess_input(image_array)  # Preprocess image for model input
    return image_array

def predict_with_models(image, models):
    # Get predictions from all models
    predictions = [model.predict(image) for model in models]
    return predictions

def ensemble_predictions(predictions, weights):
    # Assign higher weights to DenseNet and VGGNet (index 2 and 3 in model list)
    weighted_votes = np.zeros(len(predictions[0][0]))  # Initialize an array to store weighted votes
    
    # Calculate weighted votes for each prediction
    for i, prediction in enumerate(predictions):
        predicted_class = prediction.argmax()
        # Add the weight to the corresponding class vote
        weighted_votes[predicted_class] += weights[i]

    # Get final prediction based on highest weighted vote
    final_class = weighted_votes.argmax()
    confidence = weighted_votes.max() / sum(weights)  # Confidence score is based on weighted votes
    
    st.write(f"Weighted Votes: {weighted_votes}")
    return final_class, confidence

def main():
    st.title("Plant Disease Recognition")
    st.write("Upload an image to classify plant diseases using an ensemble of deep learning models.")
    
    model_paths = [
        'resnet50.keras', 
        'efficient.keras',  
        'densenet_model.keras', 
        'vgg16.keras'  
    ]
    st.write("Loading models...")
    models = [load_model(path) for path in model_paths]
    st.write("Models loaded successfully!")

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write("Processing the image...")

        image = load_img(uploaded_file)
        preprocessed_image = preprocess_image(image)

        predictions = predict_with_models(preprocessed_image, models)

        
        model_weights = [1, 1, 4, 2]  
        final_class, confidence = ensemble_predictions(predictions, model_weights)

        class_labels = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]

        predicted_label = class_labels[final_class]
        st.success(f"Predicted Class: {predicted_label}")
        st.info(f"Confidence: {confidence * 100:.2f}%")






def model_prediction(test_image):
    model=load_model("trained_model.h5")
    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    prediction=model.predict(input_arr)
    result=np.argmax(prediction)
    return result
    




st.sidebar.title("Dashboard")
app_mode=st.sidebar.selectbox("Select page", ["Home", "About", "Disease Recognition"],key="pages")
# home page
if app_mode=="Home":
    st.header("Welcome to Plant Disease Recognition App")
    image_path="home.jpg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    # 🌿 Plant Disease Prediction

## Introduction
Plant diseases can significantly reduce agricultural productivity, leading to economic losses and food scarcity. Early detection and accurate diagnosis are crucial for effective disease management. This project focuses on building a machine learning model to predict plant diseases from leaf images, helping farmers and researchers mitigate losses.

## Goals
- Detect and classify plant diseases from leaf images.
- Provide actionable information to aid in disease management.
- Improve accuracy and speed of plant disease detection using deep learning models.



## Future Work
- Extend to identify more diseases and plant species.
- Implement real-time predictions using mobile or IoT devices.



    """)

elif app_mode=="About":
    st.header("About")
    st.markdown("""
 ## Dataset
- **Source**: [New Plant Disease Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).
- **Content**: Over 87,000 labeled images covering multiple types of diseases across various plant species.
This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo. This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.

## Technology Stack
- **Python**: Core programming language.
- **TensorFlow/Keras**: Deep learning framework for model building.
- **OpenCV**: Image preprocessing.
- **Streamlit**: Web interface for user interaction.

## Model
- **Model Architecture**: Convolutional Neural Network (CNN) for image classification.
- **Loss Function**: Categorical Crossentropy.
- **Optimizer**: Adam.

## Methodology
1. **Data Collection**: Use PlantVillage dataset with labeled disease images.
2. **Data Preprocessing**:
   - Resize images for uniformity.
   - Data augmentation (flipping, rotation) to enhance model generalization.
3. **Model Training**:
   - Train CNN model on the processed dataset.
   - Evaluate using metrics like accuracy, precision, and recall.
4. **Deployment**:
   - Deploy model using Streamlit to create an accessible web app.

## Results
- Achieved an accuracy of `96.9%` on the test dataset.
- Model performs well in classifying common plant diseases.

## Conclusion
The Plant Disease Prediction model provides an effective tool for early disease diagnosis, which can help farmers take timely action and reduce crop losses.
    """)


##prediction page
elif app_mode=="Disease Recognition":
    options = ["CNN", "Ensemble Learning"]


    selected_option = st.selectbox("Choose an option:", options)
    if(selected_option == "CNN"):
    
  

      st.header("Disease Recognition")
      test_image = st.file_uploader("Choose an Image:")
      if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
      if(st.button("Predict")):
        
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
    
    else:
        main()
        
