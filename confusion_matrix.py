import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from model import build_model
import os

# Define a function to load datasets
def get_datasets(data_dir='../input/data/DATASET', train_transform=None, valid_transform=None):
    """
    Function to load training and validation datasets from a directory.
    Assumes that the directory contains subdirectories for each class.
    """
    # Apply transformations
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    if valid_transform is None:
        valid_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    # Load datasets
    train_data = ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transform)
    valid_data = ImageFolder(root=os.path.join(data_dir, 'test'), transform=valid_transform)
    
    # Get class names
    class_names = train_data.classes

    return train_data, valid_data, class_names

# Function to plot the confusion matrix.
def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """
    Plots a confusion matrix using seaborn heatmap.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

# Function to evaluate the model and generate confusion matrix.
def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on the provided dataloader and computes 
    confusion matrix.
    """
    model.eval()  # Set the model to evaluation mode.
    all_preds = []
    all_labels = []

    with torch.no_grad():  # No need to calculate gradients during evaluation.
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Get predictions from the model.
            outputs = model(images)
            _, preds = torch.max(outputs, 1)  # Get the predicted class index

            # Store predictions and actual labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute the confusion matrix.
    cm = confusion_matrix(all_labels, all_preds)
    return cm

if __name__ == '__main__':
    # Load the trained model.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(fine_tune=False, num_classes=6).to(device)

    # Load the dataset and data loader for validation.
    dataset_train, dataset_valid, dataset_classes = get_datasets()
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    print(f"[INFO]: Classes: {dataset_classes}")
    
    # Create DataLoader for validation set
    valid_loader = DataLoader(dataset_valid, batch_size=32, shuffle=False)

    # Load the model state from the best saved model.
    checkpoint = torch.load('./best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate the model and get the confusion matrix.
    cm = evaluate_model(model, valid_loader, device)

    # Plot the confusion matrix.
    plot_confusion_matrix(cm, dataset_classes, title='Confusion Matrix')
