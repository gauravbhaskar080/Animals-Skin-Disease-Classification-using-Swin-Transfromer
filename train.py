import torch                   # Imports PyTorch library for deep learning.
import argparse                # Allows parsing command-line arguments.
import torch.nn as nn          #  Imports PyTorch’s neural network module.
import torch.optim as optim    # Imports PyTorch's optimization module
import os                      # Provides a way to work with file paths

from tqdm.auto import tqdm     # Imports tqdm for progress bars.
from model import build_model
from datasets import get_datasets, get_data_loaders
from utils import save_model, save_plots, SaveBestModel

seed = 42                      # Sets the random seed for reproducibility , 
"""
By fixing the random seed, you ensure that the randomness in processes such as weight initialization, data shuffling, and any 
stochastic operation in training (e.g., dropout) is controlled, leading to consistent results across multiple runs of the same code.
"""

torch.manual_seed(seed)                        # Ensures the CPU generates the same random numbers each time.
torch.cuda.manual_seed(seed)                   # Sets the seed for CUDA operations on the GPU.
torch.backends.cudnn.deterministic = True      # Ensures that cuDNN’s operations are deterministic, avoiding random results from non-deterministic algorithms.
torch.backends.cudnn.benchmark = True          # Enables faster training by optimizing cuDNN usage. cuDNN (CUDA Deep Neural Network library) is a GPU-accelerated
                                               # library developed by NVIDIA that provides optimized primitives for deep learning applications. 

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--epochs',       # It gives flexibility to the user to set the number of training iterations.
    type=int, 
    default=10,
    help='Number of epochs to train our network for'
)
parser.add_argument(
    '-lr', '--learning-rate', # Learning rate affects how quickly or slowly the model updates weights. It's critical for model convergence.
    type=float,
    dest='learning_rate', 
    default=0.001,
    help='Learning rate for training the model'
)
parser.add_argument(
    '-b', '--batch-size',   # Batch size affects the memory usage and convergence speed during training.
    dest='batch_size',
    default=32,
    type=int
)
parser.add_argument(
    '-ft', '--fine-tune',   # If passed, it allows fine-tuning of all the layers. Without it, only the final layer(s) are trained while others are frozen.
    dest='fine_tune' ,
    action='store_true',
    help='pass this to fine tune all layers'
)
parser.add_argument(
    '--save-name',
    dest='save_name',
    default='model',
    help='file name of the final model to save'
)
args = vars(parser.parse_args()) # Parses the command-line arguments and stores them in args as a dictionary.

# Training function.
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()        # Clears old gradients
        # Forward pass.
        outputs = model(image)       # Passes the images through the model to get predictions.
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)                    # Gets the predicted class labels.
        train_running_correct += (preds == labels).sum().item()  # Accumulates the number of correct predictions.
        # Backpropagation.
        loss.backward()
        # Update the weights.
        optimizer.step()
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

# Validation function.
def validate(model, testloader, criterion, class_names):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc

if __name__ == '__main__':
    # Create a directory with the model name for outputs.
    out_dir = os.path.join('..', 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_classes = get_datasets()
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    print(f"[INFO]: Classes: {dataset_classes}")
    # Load the training and validation data loaders.
    train_loader, valid_loader = get_data_loaders(                        # Loads the data into PyTorch data loaders.
        dataset_train, dataset_valid, batch_size=args['batch_size']
    )

    # Learning_parameters.  Retrieves learning rate and epochs from the arguments.
    lr = args['learning_rate']
    epochs = args['epochs']
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")

    # Load the model.
    model = build_model(
        fine_tune=args['fine_tune'], 
        num_classes=6
    ).to(device)
    print(model)
    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())   # For each parameter p, this function returns the number of elements (or the size) of that parameter tensor (e.g., a weight matrix might have thousands of elements).
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Optimizer.
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, nesterov=True
    )
    
    """
    Nesterov accelerated gradient (NAG- advanced optimization technique) is an improvement over regular momentum, where the gradient is 
    evaluated after the current parameter update , helps make more accurate updates to the parameters 
    because it looks ahead at the next position before computing the gradient
    """
    # Loss function as cross-entropy loss for classification tasks.
    criterion = nn.CrossEntropyLoss() 

    # Initialize `SaveBestModel` class.
    save_best_model = SaveBestModel()

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                                optimizer, criterion)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  
                                                    criterion, dataset_classes)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        save_best_model(
            valid_epoch_loss, epoch, model, out_dir, args['save_name']
        )
        print('-'*50)

    # Save the trained model weights.
    save_model(epochs, model, optimizer, criterion, out_dir, args['save_name'])
    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss, out_dir)
    print('TRAINING COMPLETE')