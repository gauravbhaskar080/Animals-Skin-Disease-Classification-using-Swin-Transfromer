ANIMAL SKIN DISEASE CLASSIFICATION
==================================

Description:
------------
A deep learning application that classifies various skin conditions using a Swin Transformer model. The system can identify different skin conditions including Dermatitis, Leprosy, Lumpy Skin, and distinguish between normal skin conditions in cats, cows, and dogs.

Guide :
--------
* Dr. Bulla Rajesh

Team Members:
-------------
* Gaurav Bhaskar - S20210010080
* Mohammed Damin Khan - S20210010147
* Mohammed Faizan Ali - S20210010149


Features:
---------
* Real-time skin condition classification
* Web interface using Streamlit
* Support for multiple skin conditions
* Confusion matrix visualization for model evaluation
* Training and validation visualization
* Model performance tracking

Supported Classifications:
--------------------------
* Dermatitis
* Leprosy
* Lumpy Skin
* Normal Skin - Cat
* Normal Skin - Cow
* Normal Skin - Dog

Requirements:
-------------
Install the required packages using:
pip install -r requirements.txt

Key dependencies:
* streamlit==1.26.0
* torch>=1.12.0
* torchvision>=0.13.0
* numpy>=1.22.0
* opencv-python>=4.8.0
* Pillow>=9.2.0

Project Structure:
-----------------
app.py                   - Streamlit web application
confusion_matrix.py      - Script for generating confusion matrix
datasets.py              - Dataset loading and preprocessing
inference.py             - Model inference script
model.py                 - Model architecture definition
train.py                 - Training script
utils.py                 - Utility functions
requirements.txt         - Project dependencies

Usage Instructions:
------------------

1. Training the Model:
   Command: python train.py 

   Arguments:
   -e, --epochs         : Number of training epochs (default: 10)
   -lr, --learning-rate : Learning rate (default: 0.001)
   -b, --batch-size     : Batch size (default: 32)
   -ft, --fine-tune     : Enable fine-tuning of all layers
   --save-name          : Name for the saved model (default: 'model')

2. Running the Web Interface:
   Command: streamlit run app.py
   The application will be accessible through your web browser.

3. Running Inference:
   Command: python inference.py 

4. Generating Confusion Matrix:
   Command: python confusion_matrix.py

Data Organization:
-----------------
The dataset should be organized as follows:

input/
    data/
        DATASET/
            train/
                Dermatitis/
                Leprosy/
                Lumpy Skin/
                Normal Skin - Cat/
                Normal Skin - Cow/
                Normal Skin - Dog/
            test/
                Dermatitis/
                Leprosy/
                Lumpy Skin/
                Normal Skin - Cat/
                Normal Skin - Cow/
                Normal Skin - Dog/

Model Architecture:
------------------
* Uses Swin Transformer model (swin_t)
* Pre-trained weights from ImageNet
* Custom classification head for 6 classes
* Optional fine-tuning of all layers

Output Files:
------------
The training process generates:
* Model checkpoints (model.pth and best_model.pth)
* Training/validation accuracy plots (accuracy.png)
* Training/validation loss plots (loss.png)
* Inference results in outputs/inference_results/

Results:
--------

1. Web Interface Classifications:

* Cow Normal 

* Lumpy Skin

* Dog Normal 

* Dermatitis

* Cat Normal 

* Leprosy

2. Loss and Accuracy Graphs:

3. Confusion Matrix



Additional Notes:
----------------
* The model uses CUDA if available, otherwise falls back to CPU
* Images are automatically resized to 224x224 pixels
* Data augmentation is applied during training
* Best model weights are automatically saved based on validation loss




