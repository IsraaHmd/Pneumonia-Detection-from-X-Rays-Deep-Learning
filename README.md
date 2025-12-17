# Pneumonia Detection from Chest X-Rays

A deep learning project that classifies chest X-ray images as either Normal or Pneumonia, demonstrating how AI can support medical diagnosis.

## Project Overview

Pneumonia is a lung infection that causes inflammation and fluid buildup in the air sacs, with symptoms ranging from mild to life-threatening. This project builds a convolutional neural network (CNN) to automatically detect pneumonia from chest X-ray images.

## Goal

Develop and compare deep learning models that can accurately classify chest X-rays into two categories:
- **NORMAL**: Healthy lungs
- **PNEUMONIA**: Infected lungs

## Dataset

**Source**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) - Kaggle

- **Total Images**: 5,863
- **Classes**: NORMAL, PNEUMONIA
- **Splits**: Training, Validation, Test sets

## Technologies & Libraries

- **Python 3.x**
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Image processing
- **NumPy** - Numerical operations
- **Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Visualization
- **scikit-learn** - Metrics and evaluation

## Project Structure

```
├── Data Exploration
│   ├── Sample visualization
│   ├── Class distribution analysis
│   └── Pixel intensity inspection
│
├── Data Preprocessing
│   ├── Image resizing (224x224)
│   ├── Grayscale conversion
│   ├── Pixel normalization
│   └── Data augmentation
│
├── Model Development
│   ├── Basic CNN (4 convolutional blocks)
│   └── Transfer Learning (DenseNet121)
│
└── Evaluation
    ├── Accuracy, Precision, Recall, F1-Score
    ├── Confusion Matrix
    └── Training/Validation curves
```

## Models

### 1. Basic CNN
- **Architecture**: 4 convolutional blocks with batch normalization and max pooling
- **Input**: 224x224 grayscale images
- **Output**: Binary classification (sigmoid activation)

### 2. Transfer Learning (DenseNet121)
- **Base Model**: Pre-trained DenseNet121 (ImageNet weights)
- **Input**: 224x224 RGB images (grayscale converted to 3-channel)
- **Approach**: Frozen base + custom classification head

## Key Features

- **Class Imbalance Handling**: Balanced validation set creation
- **Data Augmentation**: Width/height shifts, zoom transformations
- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score
- **Visual Analysis**: Confusion matrices and training curves

## Evaluation Metrics

The models are evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: Correct positive predictions
- **Recall**: Ability to find all positive cases
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

## Usage

1. **Setup Dataset**: Place the dataset in `/data/` directory with `train/`, `test/`, and `val/` folders

2. **Install Dependencies**:
```bash
pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn
```

3. **Run the Notebook**: Execute cells sequentially to train and evaluate models

## Notes

- Images are preprocessed to 224x224 pixels
- Grayscale conversion reduces computational cost (1 channel vs 3)
- Data augmentation helps address class imbalance
- Early stopping restores best weights based on validation loss

## Results

Both models provide detailed performance metrics including:
- Test set accuracy and loss
- Precision, recall, and F1-scores
- Confusion matrices showing true/false positives and negatives
- Training history plots comparing training vs validation performance

## References

- Harvard Health Publishing. (2024). Pneumonia: Symptoms, causes, and treatment.
- DataCamp. Complete guide to data augmentation.
- Kaggle Dataset: chest-xray-pneumonia

Note: This project is for educational and demonstration purposes. Medical diagnosis should always be performed by qualified healthcare professionals.

---

**Note**: This project is for educational and demonstration purposes. Medical diagnosis should always be performed by qualified healthcare professionals.
