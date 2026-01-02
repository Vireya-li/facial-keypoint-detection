# Facial Keypoint Detection

A facial keypoint detection project based on Convolutional Neural Networks, referencing [Daniel Nouri's classic tutorial](https://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/), re-implemented using the modern PyTorch framework.

## Project Overview

This project aims to detect 15 keypoint positions in facial images, including:
- Eye centers
- Eye corners (inner and outer)
- Eyebrow endpoints
- Nose tip
- Mouth corners
- Lip center

![Example](https://kaggle2.blob.core.windows.net/competitions/kaggle/3486/media/face1_with_keypoints.png)

## Features

- ✅ **Multiple Network Architectures**: From simple fully connected networks to advanced convolutional neural networks
- ✅ **Data Augmentation**: Horizontal flip augmentation to effectively expand training data
- ✅ **Dynamic Learning Rate**: Learning rate and momentum automatically adjust during training
- ✅ **Dropout Regularization**: Prevents overfitting
- ✅ **Early Stopping**: Automatically stops training and saves the best model
- ✅ **Specialist Models**: Dedicated models trained for different keypoint groups
- ✅ **Pretrained Transfer**: Supports weight transfer to accelerate specialist model training

## Project Structure

```
facial_keypoint_detection/
├── data/                    # Data directory
│   ├── training.csv         # Training data
│   └── test.csv             # Test data
├── data_loader.py           # Data loading module
├── models.py                # Neural network model definitions
├── train.py                 # Training script
├── specialists.py           # Specialist models
├── kfkd.py                  # Main entry point
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Data

1. Visit [Kaggle Facial Keypoints Detection Competition](https://www.kaggle.com/c/facial-keypoints-detection/data)
2. Download `training.csv` and `test.csv`
3. Place the files in the `data/` directory

## Usage

### Train Base Model

```bash
# Train with default parameters
python kfkd.py train

# Custom parameters
python kfkd.py train --epochs 500 --lr 0.01 --model advanced
```

### Train Specialist Models

Specialist models are trained specifically for different keypoint groups and usually achieve better results:

```bash
# First train the base model, then train specialist models
python kfkd.py train
python kfkd.py train_specialists
```

### Generate Predictions

```bash
python kfkd.py predict --model-path model.pth
```

### Visualize Results

```bash
python kfkd.py visualize --model-path model.pth
```

## Model Architectures

### 1. SimpleNet
- Single hidden layer fully connected network
- Suitable for quick validation

### 2. ConvNet
- 3 convolutional layers + 3 fully connected layers
- Standard CNN architecture

### 3. ConvNetDropout
- Adds Dropout on top of ConvNet
- Effectively prevents overfitting

### 4. ConvNetAdvanced
- 4 convolutional layers + Batch Normalization
- Deeper fully connected layers
- Best performance

## Core Concepts

### Data Augmentation
Increase training data by horizontally flipping images. When flipping, note that:
- x coordinates need to be negated
- Symmetric keypoints need to be swapped (e.g., left and right eyes)

### Dynamic Adjustment
During training:
- **Learning Rate**: Gradually decreases from 0.01 to 0.001
- **Momentum**: Gradually increases from 0.9 to 0.999

### Specialist Models
Due to different keypoints having different numbers of training samples:
- Eye centers: ~7000 samples
- Eye corners: ~2000 samples

Therefore, dedicated "specialist" models can be trained to handle specific keypoint groups.

### Pretrained Transfer
Using trained base model weights to initialize specialist models can:
- Accelerate convergence
- Improve generalization

## Expected Results

| Model | Validation RMSE (pixels) |
|-------|--------------------------|
| SimpleNet | ~4.0 |
| ConvNet | ~3.5 |
| ConvNetDropout | ~3.0 |
| ConvNetAdvanced | ~2.5 |
| Specialist Models (pretrained) | ~2.2 |

## References

- [Original Tutorial](https://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/)
- [Kaggle Competition Page](https://www.kaggle.com/c/facial-keypoints-detection)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)

## License

MIT License
