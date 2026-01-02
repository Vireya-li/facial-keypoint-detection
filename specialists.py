"""
Specialist Models - Dedicated models for different keypoint groups
Reference: https://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/

Core idea of specialist models:
- Different keypoints have different numbers of training samples
- Train dedicated small models to handle specific keypoint groups
- Use pretrained weights to accelerate training
"""

import os
import pickle
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import load2d, FacialKeypointsDataset
from models import ConvNetAdvanced
from train import train, EarlyStopping, train_epoch, validate


# Specialist model settings
# Each dictionary contains:
#   - columns: Keypoint columns this specialist is responsible for
#   - flip_indices: Index pairs to swap during horizontal flip
SPECIALIST_SETTINGS = [
    # Specialist 1: Eye centers
    {
        'columns': (
            'left_eye_center_x', 'left_eye_center_y',
            'right_eye_center_x', 'right_eye_center_y',
        ),
        'flip_indices': ((0, 2), (1, 3)),
    },
    
    # Specialist 2: Nose tip
    {
        'columns': (
            'nose_tip_x', 'nose_tip_y',
        ),
        'flip_indices': (),
    },
    
    # Specialist 3: Mouth corners
    {
        'columns': (
            'mouth_left_corner_x', 'mouth_left_corner_y',
            'mouth_right_corner_x', 'mouth_right_corner_y',
        ),
        'flip_indices': ((0, 2), (1, 3)),
    },
    
    # Specialist 4: Lip center
    {
        'columns': (
            'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y',
        ),
        'flip_indices': (),
    },
    
    # Specialist 5: Eye corners
    {
        'columns': (
            'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
            'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
            'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
            'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
        ),
        'flip_indices': ((0, 2), (1, 3), (4, 6), (5, 7)),
    },
    
    # Specialist 6: Eyebrows
    {
        'columns': (
            'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
            'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
            'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
            'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
        ),
        'flip_indices': ((0, 2), (1, 3), (4, 6), (5, 7)),
    },
]


def load_pretrained_model(model_path):
    """Load pretrained model"""
    checkpoint = torch.load(model_path, map_location='cpu')
    model = ConvNetAdvanced(output_dim=30)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def transfer_weights(source_model, target_model):
    """
    Transfer weights from source model to target model
    
    Only transfers matching layers (except the final output layer)
    """
    source_dict = source_model.state_dict()
    target_dict = target_model.state_dict()
    
    # Find layers that can be transferred
    transferred = []
    for name, param in source_dict.items():
        if name in target_dict and 'fc3' not in name:  # Don't transfer the last layer
            if target_dict[name].shape == param.shape:
                target_dict[name] = param.clone()
                transferred.append(name)
    
    target_model.load_state_dict(target_dict)
    print(f"Transferred {len(transferred)} layer weights")
    
    return target_model


def train_specialist(setting, pretrain_model=None, n_epochs=500, device='cuda'):
    """
    Train a single specialist model
    
    Args:
        setting: Specialist settings dictionary
        pretrain_model: Pretrained model (optional)
        n_epochs: Number of training epochs
        device: Computing device
    """
    cols = setting['columns']
    flip_indices = setting['flip_indices']
    
    print(f"\n{'='*60}")
    print(f"Training specialist model: {cols}")
    print(f"{'='*60}")
    
    # Load data for this specialist
    X, y = load2d(cols=cols)
    output_dim = y.shape[1]
    
    print(f"Number of samples: {len(X)}, Output dimension: {output_dim}")
    
    # Split dataset
    n_train = int(len(X) * 0.8)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]
    
    # Create data loaders
    train_dataset = FacialKeypointsDataset(X_train, y_train, augment=True,
                                           flip_indices=flip_indices)
    val_dataset = FacialKeypointsDataset(X_val, y_val, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Create specialist model
    model = ConvNetAdvanced(output_dim=output_dim)
    
    # If pretrained model exists, transfer weights
    if pretrain_model is not None:
        print("Initializing with pretrained weights...")
        model = transfer_weights(pretrain_model, model)
    
    # Adjust number of epochs based on sample count
    adjusted_epochs = min(n_epochs, int(1e6 / len(X)))
    print(f"Adjusted number of epochs: {adjusted_epochs}")
    
    # Train model
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=adjusted_epochs,
        lr_start=0.01 if pretrain_model is None else 0.003,
        lr_end=0.001,
        patience=50,
        device=device
    )
    
    return model, history


def fit_specialists(pretrain_path=None, save_path='specialists.pkl', device='cuda'):
    """
    Train all specialist models
    
    Args:
        pretrain_path: Pretrained model path (optional)
        save_path: Save path
        device: Computing device
    """
    # Load pretrained model
    pretrain_model = None
    if pretrain_path and os.path.exists(pretrain_path):
        print(f"Loading pretrained model: {pretrain_path}")
        pretrain_model = load_pretrained_model(pretrain_path)
    
    # Train each specialist
    specialists = OrderedDict()
    
    for i, setting in enumerate(SPECIALIST_SETTINGS):
        print(f"\nSpecialist {i+1}/{len(SPECIALIST_SETTINGS)}")
        
        model, history = train_specialist(
            setting=setting,
            pretrain_model=pretrain_model,
            device=device
        )
        
        specialists[setting['columns']] = {
            'model': model.cpu().state_dict(),
            'history': history,
            'columns': setting['columns'],
            'output_dim': len(setting['columns']),
        }
    
    # Save all specialist models
    with open(save_path, 'wb') as f:
        pickle.dump(specialists, f)
    
    print(f"\nAll specialist models saved to: {save_path}")
    
    return specialists


def load_specialists(path='specialists.pkl'):
    """Load specialist models"""
    with open(path, 'rb') as f:
        specialists = pickle.load(f)
    
    loaded_models = OrderedDict()
    for cols, data in specialists.items():
        model = ConvNetAdvanced(output_dim=data['output_dim'])
        model.load_state_dict(data['model'])
        loaded_models[cols] = model
    
    return loaded_models


def predict_with_specialists(specialists, X, device='cuda'):
    """
    Make predictions using specialist models
    
    Args:
        specialists: Specialist models dictionary
        X: Input images (N, 1, 96, 96)
        device: Computing device
    
    Returns:
        predictions: Dictionary of predictions for all keypoints
    """
    predictions = {}
    X_tensor = torch.FloatTensor(X).to(device)
    
    for cols, model in specialists.items():
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            pred = model(X_tensor).cpu().numpy()
        
        # Restore coordinates
        pred = pred * 48 + 48
        
        # Store predictions for each keypoint
        for i, col in enumerate(cols):
            predictions[col] = pred[:, i]
    
    return predictions


if __name__ == "__main__":
    import torch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train specialist models
    # If base model is already trained, pass the pretrained model path
    specialists = fit_specialists(
        pretrain_path='model.pth',  # If exists
        save_path='specialists.pkl',
        device=device
    )
    
    print("\nTraining complete!")
