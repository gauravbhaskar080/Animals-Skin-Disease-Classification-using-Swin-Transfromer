import streamlit as st
import torch
import numpy as np
import cv2
import os
import torch.nn.functional as F #e.g., softmax
import torchvision.transforms as transforms
import glob #Finds files and directories based on a pattern
import argparse
import pathlib
from PIL import Image
from model import build_model

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-w', '--weights', 
    # default='../outputs/best_model.pth',
    default='./best_model.pth',
    help='path to the model weights',
)
args = vars(parser.parse_args())

# Constants and other configurations.
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IMAGE_RESIZE = 224
CLASS_NAMES = ['Dermatitis','Normal Skin - Dog', 'Lumpy Skin', 'Normal Skin - Cow', 'Leprosy', 'Normal Skin - Cat']

# Validation transforms
def get_test_transform(image_size):
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
        # Convert to a tensor and normalize with mean and std values (common for models like ResNet).
    ])
    return test_transform

# Annotates the original image with the predicted class label.
def annotate_image(output_class, orig_image):
    class_name = CLASS_NAMES[int(output_class)]
    cv2.putText(
        orig_image, 
        f"{class_name}", 
        (5, 35), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1.5, 
        (0, 0, 255), 
        2, 
        lineType=cv2.LINE_AA
    )
    return orig_image

# Inference refers to the process of using a trained machine learning model to make predictions on new, unseen data.
# Performs inference on a test image and annotates the result.
def inference(model, testloader, device, orig_image):
    """
    Function to run inference.

    :param model: The trained model.
    :param testloader: The test data loader.
    :param DEVICE: The computation device.
    """
    model.eval()
    counter = 0
    with torch.no_grad():
        counter += 1
        image = testloader
        image = image.to(device)

        # Forward pass.
        outputs = model(image)
    # Softmax probabilities : mathematical function used in machine learning, particularly in classification tasks, to convert raw model outputs (logits) into probabilities.
    predictions = F.softmax(outputs, dim=1).cpu().numpy()
    # Predicted class number : argmax is a function that returns the index of the maximum value in an array or tensor.
    output_class = np.argmax(predictions)
    # Show and save the results.
    result = annotate_image(output_class, orig_image)
    return result

if __name__ == '__main__':
    weights_path = pathlib.Path(args['weights'])
    infer_result_path = os.path.join(
        '..', 'outputs', 'inference_results'
    )
    os.makedirs(infer_result_path, exist_ok=True)

    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))

    # Load the model.
    model = build_model(
        fine_tune=False, 
        num_classes=6
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Retrieves all image paths from the inference_data directory using glob.glob()
    all_image_paths = glob.glob(os.path.join('..', 'input', 'inference_data', '*'))

    transform = get_test_transform(IMAGE_RESIZE)

    for i, image_path in enumerate(all_image_paths):
        print(f"Inference on image: {i+1}")    
        try:
            image = cv2.imread(image_path)
            orig_image = image.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transform(image)
            image = torch.unsqueeze(image, 0)
            result = inference(
                model, 
                image,
                DEVICE,
                orig_image
            )
            # Save the image to disk.
            image_name = image_path.split(os.path.sep)[-1]
            cv2.imshow('Image', result)
            cv2.waitKey(1)
            cv2.imwrite(
                os.path.join(infer_result_path, image_name), result
            )
        except UnidentifiedImageError as e:
            print(f"Unidentified image: {image_path}. Skipping...")
            continue

"""
The script performs image classification using a pre-trained PyTorch 
model. It loads images from a directory, preprocesses them (resize, 
normalize), runs inference, annotates the image with the predicted class,
and saves the result. The code efficiently handles GPU support and command-line 
argument parsing.
"""