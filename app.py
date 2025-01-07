import streamlit as st
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from model import build_model

# Constants
CLASS_NAMES = ['Dermatitis', 'Leprosy' ,'Lumpy Skin', 'Normal Skin - Cat', 'Normal Skin - Cow' ,'Normal Skin - Dog']
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 224
# MODEL_WEIGHTS_PATH = '../outputs/best_model.pth'
MODEL_WEIGHTS_PATH = './best_model.pth'

# Load the trained model
@st.cache_resource
def load_model():
    model = build_model(fine_tune=False, num_classes=len(CLASS_NAMES))
    checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# Define the preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    image = torch.unsqueeze(image, 0)  # Add batch dimension
    return image.to(DEVICE)

# Annotate the image with the predicted class
def annotate_image(image, prediction):
    annotated_image = np.array(image)
    cv2.putText(
        annotated_image,
        f"Prediction: {CLASS_NAMES[prediction]}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        lineType=cv2.LINE_AA
    )
    return annotated_image

# Streamlit app UI
st.title("Skin Condition Classifier")
st.write("Upload an image of the skin to classify its condition.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load the image
    image = Image.open(uploaded_file)
    
     # Create two columns
    col1, col2 = st.columns(2)

    # Display the uploaded image in the first column
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
    # st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Processing..."):
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Perform inference
        with torch.no_grad():
            outputs = model(processed_image)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()
            prediction = np.argmax(probabilities)
        
        # Annotate the image
        annotated_image = annotate_image(image, prediction)

    # Display the annotated image and prediction in the second column
    with col2:
        st.image(annotated_image, caption="Annotated Image", use_column_width=True)
        st.success(f"Prediction: {CLASS_NAMES[prediction]}")
