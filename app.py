import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

# Define the labels mapping (integer to bear type)
df_labels = {
    'Black': 0,
    'Grizzly': 1,
    'Panda': 2,
    'Polar': 3,
    'Teddy': 4
}

# Reverse the dictionary (from integer to bear type)
label_to_bear = {v: k for k, v in df_labels.items()}

@st.cache_resource
def load_bear_model():
    return load_model('bear_model/model.h5')

model = load_bear_model()

# Streamlit UI for uploading an image
st.title('Bear Type Classification')
st.write("Upload an image of a bear to classify its type.")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image with a smaller size
    img = load_img(uploaded_file, target_size=(224, 224))
    
    # Option 1: Show the image at a smaller size by specifying the width (in pixels)
    st.image(img, caption="Uploaded Image", width=300)  # Resize width to 300 pixels
    
    # Option 2: Alternatively, set the height of the image
    # st.image(img, caption="Uploaded Image", height=300)  # Resize height to 300 pixels
    
    # Preprocess the image for prediction
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize if necessary

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)

    # Map prediction to bear type
    predicted_bear = label_to_bear[predicted_class[0]]

    # Display the result
    st.write(f"Predicted bear type: {predicted_bear}")

    # Optionally, display the predicted image with the prediction as title
    # Resize the predicted image before displaying
    fig, ax = plt.subplots(figsize=(5, 5))  # You can adjust this as needed
    ax.imshow(img)
    ax.set_title(f"Predicted Bear: {predicted_bear}")
    ax.axis('off')
    
    # Show the predicted image at a smaller size in Streamlit
    st.pyplot(fig)
