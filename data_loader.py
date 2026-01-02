"""
Data Loading Module - Kaggle Facial Keypoint Detection
Reference: https://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
"""

import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset, DataLoader


# Data file paths (need to download from Kaggle)
FTRAIN = 'data/training.csv'
FTEST = 'data/test.csv'


def load(test=False, cols=None):
    """
    Load data
    
    Args:
        test: If True, load test data, otherwise load training data
        cols: Optional, specify target columns to load
        
    Returns:
        X: Image data, shape (n_samples, 9216), values in range [0, 1]
        y: Target values, shape (n_samples, n_targets), values in range [-1, 1]
    """
    fname = FTEST if test else FTRAIN
    df = pd.read_csv(os.path.expanduser(fname))
    
    # Image column contains space-separated pixel values, convert to numpy array
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    
    if cols:  # Get subset of columns
        df = df[list(cols) + ['Image']]
    
    print(df.count())  # Print the number of values for each column
    df = df.dropna()   # Drop rows with missing values
    
    X = np.vstack(df['Image'].values) / 255.0  # Scale pixel values to [0, 1]
    X = X.astype(np.float32)
    
    if not test:  # Only training data has target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # Scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # Shuffle training data
        y = y.astype(np.float32)
    else:
        y = None
    
    return X, y


def load2d(test=False, cols=None):
    """
    Load data in 2D format (for convolutional networks)
    
    Returns:
        X: Image data, shape (n_samples, 1, 96, 96)
        y: Target values
    """
    X, y = load(test=test, cols=cols)
    X = X.reshape(-1, 1, 96, 96)  # (samples, channels, height, width)
    return X, y


class FacialKeypointsDataset(Dataset):
    """
    PyTorch Dataset class for facial keypoint data
    """
    
    def __init__(self, X, y=None, augment=False, flip_indices=None):
        """
        Args:
            X: Image data
            y: Target values (optional)
            augment: Whether to perform data augmentation
            flip_indices: Keypoint index pairs to swap during horizontal flip
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
        self.augment = augment
        self.flip_indices = flip_indices if flip_indices else []
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = self.X[idx]
        
        if self.y is not None:
            y = self.y[idx]
            
            # Data augmentation: random horizontal flip
            if self.augment and np.random.random() > 0.5:
                X = torch.flip(X, dims=[2])  # Horizontally flip image
                y = y.clone()
                
                # Flip x coordinates
                y[0::2] = -y[0::2]
                
                # Swap corresponding keypoints
                for a, b in self.flip_indices:
                    y[a], y[b] = y[b].clone(), y[a].clone()
            
            return X, y
        else:
            return X


def get_data_loaders(batch_size=128, augment=True, cols=None, flip_indices=None):
    """
    Get training and validation data loaders
    
    Args:
        batch_size: Batch size
        augment: Whether to perform data augmentation
        cols: Target columns to load
        flip_indices: Index pairs to swap during flip
        
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
    """
    X, y = load2d(cols=cols)
    
    # Split training and validation sets (80% / 20%)
    n_train = int(len(X) * 0.8)
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    train_dataset = FacialKeypointsDataset(X_train, y_train, augment=augment, 
                                           flip_indices=flip_indices)
    val_dataset = FacialKeypointsDataset(X_val, y_val, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


# Flip index pairs for all 15 keypoints
# When image is horizontally flipped, left and right eye coordinates need to be swapped
FLIP_INDICES = [
    (0, 2), (1, 3),     # left_eye_center <-> right_eye_center
    (4, 8), (5, 9),     # left_eye_inner_corner <-> right_eye_inner_corner
    (6, 10), (7, 11),   # left_eye_outer_corner <-> right_eye_outer_corner
    (12, 16), (13, 17), # left_eyebrow_inner_end <-> right_eyebrow_inner_end
    (14, 18), (15, 19), # left_eyebrow_outer_end <-> right_eyebrow_outer_end
    (22, 24), (23, 25), # mouth_left_corner <-> mouth_right_corner
]


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")
    try:
        X, y = load()
        print(f"\nX.shape = {X.shape}; X.min = {X.min():.3f}; X.max = {X.max():.3f}")
        print(f"y.shape = {y.shape}; y.min = {y.min():.3f}; y.max = {y.max():.3f}")
    except FileNotFoundError:
        print("\nPlease download the Kaggle dataset and place it in the data/ directory:")
        print("1. Visit https://www.kaggle.com/c/facial-keypoints-detection/data")
        print("2. Download training.csv and test.csv")
        print("3. Place the files in the data/ directory")
