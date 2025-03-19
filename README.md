# MNIST Digit Classification using CNN (PyTorch)

## Overview

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset. The dataset is provided in CSV format and contains pixel values of 28x28 grayscale images. The model is trained to achieve high accuracy in digit recognition.

## Dataset

The dataset used for this project can be downloaded from Kaggle:
[MNIST in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

It consists of:

- `mnist_train.csv`: Training dataset
- `mnist_test.csv`: Test dataset

Each row represents an image, with the first column being the label (0-9) and the remaining 784 columns representing the pixel values.

## Model Architecture

The CNN model consists of the following layers:

1. **Conv2D (32 filters, 3x3 kernel, ReLU activation) → MaxPooling (2x2)**
2. **Conv2D (64 filters, 3x3 kernel, ReLU activation) → MaxPooling (2x2)**
3. **Flatten layer**
4. **Fully connected layer (128 neurons, ReLU activation)**
5. **Dropout (0.5)**
6. **Output layer (10 neurons, softmax activation for classification)**

This architecture efficiently captures spatial features and patterns from the handwritten digits.

## Installation

To run this project, you need Python 3 and the following dependencies:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn
```

Alternatively, if you are using Kaggle, the required libraries are pre-installed.

## Running the Model

1. Clone this repository:

```bash
git clone https://github.com/yourusername/mnist-cnn-pytorch.git
cd mnist-cnn-pytorch
```

2. Download and place the dataset in the appropriate directory (`/kaggle/input/mnist-in-csv/` or the local folder).

3. Run the script:

```bash
python train.py
```

The script will:

- Load the dataset
- Train the CNN model
- Evaluate the model on test data
- Save the trained model (`mnist_cnn_model.pth`)
- Generate visualizations (accuracy plot, confusion matrix, sample predictions)

## Model Training and Evaluation

- The model is trained using **CrossEntropyLoss** and optimized with **Adam optimizer**.
- The accuracy is tracked across epochs.
- A confusion matrix is generated to analyze misclassifications.
- Predictions on test samples are visualized.

## Results

- The trained model achieves high accuracy on the MNIST dataset.
- Predictions and metrics are saved for further analysis.

## Visualizations

The script generates:

- **Accuracy plot**: Training vs. test accuracy over epochs.
- **Confusion matrix**: Helps in understanding misclassifications.
- **Sample predictions**: Displays images along with predicted vs. true labels.

## Saving and Loading the Model

To save the model:

```python
torch.save(model.state_dict(), 'mnist_cnn_model.pth')
```

To load and use the trained model:

```python
model = CNN()
model.load_state_dict(torch.load('mnist_cnn_model.pth'))
model.eval()
```

## Future Improvements

- Hyperparameter tuning for better accuracy
- Data augmentation for improved generalization
- Deploying the model using Flask or FastAPI

## License

This project is licensed under the MIT License.

## Acknowledgments

- [PyTorch](https://pytorch.org/)
- [Kaggle MNIST Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
