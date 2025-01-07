from torchvision import models

import torch.nn as nn


# Update num_classes to match the actual number of classes
def build_model(fine_tune=True, num_classes=6):  # fine_tune determines if all layers are trainable
    model = models.swin_t(weights='DEFAULT')     # Loads a pre-trained Swin Transformer model with default weights. swin_t refers to the Swin Transformer model variant.
    print(model)
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():   
            params.requires_grad = True           # Sets requires_grad to True for all parameters in the model, making them trainable.
    if not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False          # Sets requires_grad to False for all parameters in the model, freezing them and making them untrainable.

    model.head = nn.Linear(
        in_features=768,           # Defines a fully connected layer with 768 input features (from the Swin model)
        out_features=num_classes,  # Update out_features to num_classes
        bias=True
    )
    return model


if __name__ == '__main__':
    model = build_model()
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())    # Computes the total number of parameters in the model by summing the number of elements in each parameter tensor.
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    

"""
The code builds a Swin Transformer model, optionally fine-tunes or 
freezes layers based on the fine_tune parameter, and replaces the final 
classification layer to match the number of output classes. It then prints
the model's architecture and the total number of parameters, including 
trainable parameters.
"""