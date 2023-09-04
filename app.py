import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import pyttsx3

# Load the trained model
model = tf.keras.models.load_model('model_with_regularization.keras')

# Prepare a list of symbols (the order should correspond to the class indices used during training)
symbols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Streamlit UI
st.title("Sign Language Recognition")
st.write("Upload an image and let the model predict the sign!")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Make prediction and display result
if uploaded_image is not None:
    img = Image.open(uploaded_image)
    img = img.resize((100, 100))
    img_array = np.array(img) / 255.0
    roi = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(roi)

    # Get the predicted sign label
    predicted_label = np.argmax(prediction)
    predicted_sign = symbols[predicted_label]

    # Display the prediction result on the uploaded image
    img = np.array(img)
    cv2.putText(img, predicted_sign, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Resize the image for better visualization
    resized_img = cv2.resize(img, (800, 600))

    # Display the resized image with the prediction result
    st.image(resized_img, caption=f"Predicted Sign: {predicted_sign}", channels="RGB")

    # Convert predicted sign to speech
    engine.say(f"The predicted sign is {predicted_sign}")
    engine.runAndWait()
