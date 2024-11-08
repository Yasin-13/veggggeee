# importing Libraries
import os
import uuid
import streamlit as st
import numpy as np
import urllib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Loading the model
model = load_model('veggiesClassification_model.keras')

class_map = {0: 'Bean', 1: 'Bitter Gourd', 2: 'Bottle Gourd', 3: 'Brinjal', 4: 'Broccoli', 5: 'Cabbage', 6: 'Capsicum', 7: 'Carrot', 8: 'Cauliflower', 9: 'Cucumber', 10: 'Papaya', 11: 'Potato', 12: 'Pumpkin', 13: 'Radish', 14: 'Tomato'}

# Function to predict the vegetable
def predict(image, model):
    test_image = image.resize((224, 224))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    predictions = model.predict(test_image)
    sorted_labels = np.argsort(predictions[0])[::-1]
    top3_indices = sorted_labels[:3]

    prob_result = []
    class_result = []
    for i in top3_indices:
        prob_result.append((predictions[0][i] * 100).round(2))
        class_result.append(class_map[i])

    return class_result, prob_result

# Streamlit app
st.title("Vegetable Classification")


# Information text
st.write("The model is trained using convolutional neural networks and can classify an image into following vegetables:")
st.write("Bean, Bitter Gourd, Bottle Gourd, Brinjal, Broccoli, Cabbage, Capsicum, Carrot, Cauliflower, Cucumber, Papaya, Potato, Pumpkin, Radish, Tomato.")
st.write("")

# Image upload
uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([3.5, 3])
    # Image displaying
    with col1:
        st.write("")
        st.write("")
        st.image(image, caption="Uploaded Image", width=300)
    # Predictions Displaying
    with col2:
        st.write("")
        st.write("Prediction Results:")
        class_result, prob_result = predict(image, model)

        # Displaying the results in table format
        data = {'Vegeatble': class_result, 'Probability (%)': prob_result}
        st.markdown(
                f"""
                <style>
                .large-table {{
                    width: 500px;  /* Adjust width as needed */
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
        st.table(data)

# URL upload
st.write("")
image_url = st.text_input("Or Enter an image URL")

if image_url:
    try:
        resource = urllib.request.urlopen(image_url)
        unique_filename = str(uuid.uuid4())
        filename = unique_filename + ".jpg"
        img_path = os.path.join("uploaded_images", filename)
        with open(img_path, "wb") as output:
            output.write(resource.read())

        image = Image.open(img_path)

        # Image displaying
        col1, col2 = st.columns([3.5, 3])
        with col1:
            st.write("")
            st.write("")
            st.image(image, caption="Uploaded Image", width=300)
        # Predictions Displaying
        with col2:
            st.write("")
            st.write("Prediction Results:")
            class_result, prob_result = predict(image, model)

            # Displaying the results in table format
            data = {'Vegetable': class_result, 'Probability (%)': prob_result}
            st.markdown(
                f"""
                <style>
                .large-table {{
                    width: 500px;  /* Adjust width as needed */
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
            st.table(data)

    except Exception as e:
        st.error(f"Unable to access image from the provided link: {str(e)}")
