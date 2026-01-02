#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Facial Keypoint Detection - Main Entry
Reference: https://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/

Usage:
    python kfkd.py train              # Train base model
    python kfkd.py train_specialists  # Train specialist models
    python kfkd.py predict            # Generate predictions
    python kfkd.py visualize          # Visualize prediction results
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load2d, FacialKeypointsDataset, FLIP_INDICES
from models import get_model, ConvNetAdvanced
from train import train, plot_learning_curves, visualize_predictions
from torch.utils.data import DataLoader


def cmd_train(args):
    """Train base model"""
    print("=" * 60)
    print("Training Facial Keypoint Detection Model")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    X, y = load2d()
    
    # Split dataset
    n_train = int(len(X) * 0.8)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Create data loaders
    train_dataset = FacialKeypointsDataset(X_train, y_train, augment=True,
                                           flip_indices=FLIP_INDICES)
    val_dataset = FacialKeypointsDataset(X_val, y_val, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)
    
    # Create model
    print(f"\nCreating model: {args.model}")
    model = get_model(args.model, output_dim=30)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Train
    print("\nStarting training...")
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.epochs,
        lr_start=args.lr,
        lr_end=args.lr / 10,
        patience=args.patience,
        device=device
    )
    
    # Save model
    save_path = args.output or 'model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'model_type': args.model,
    }, save_path)
    print(f"\nModel saved to: {save_path}")
    
    # Plot learning curves
    plot_learning_curves(history, 'learning_curves.png')
    
    # Visualize predictions
    visualize_predictions(model, X_val, y_val, device=device)


def cmd_train_specialists(args):
    """Train specialist models"""
    from specialists import fit_specialists
    
    print("=" * 60)
    print("Training Specialist Models")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    specialists = fit_specialists(
        pretrain_path=args.pretrain,
        save_path=args.output or 'specialists.pkl',
        device=device
    )
    
    print("\nSpecialist model training complete!")


def cmd_predict(args):
    """Generate predictions"""
    print("=" * 60)
    print("Generating Predictions")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"Loading model: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    model = ConvNetAdvanced(output_dim=30)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load test data
    print("Loading test data...")
    X_test, _ = load2d(test=True)
    
    print(f"Number of test samples: {len(X_test)}")
    
    # Generate predictions
    print("Generating predictions...")
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X_test), 128):
            batch = torch.FloatTensor(X_test[i:i+128]).to(device)
            pred = model(batch).cpu().numpy()
            predictions.append(pred)
    
    predictions = np.vstack(predictions)
    
    # Restore coordinates
    predictions = predictions * 48 + 48
    
    # Create submission file
    columns = [
        'left_eye_center_x', 'left_eye_center_y',
        'right_eye_center_x', 'right_eye_center_y',
        'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
        'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
        'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
        'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
        'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
        'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
        'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
        'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
        'nose_tip_x', 'nose_tip_y',
        'mouth_left_corner_x', 'mouth_left_corner_y',
        'mouth_right_corner_x', 'mouth_right_corner_y',
        'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
        'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y',
    ]
    
    df = pd.DataFrame(predictions, columns=columns)
    
    output_path = args.output or 'predictions.csv'
    df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")


def cmd_visualize(args):
    """Visualize prediction results"""
    print("=" * 60)
    print("Visualizing Predictions")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location=device)
    model = ConvNetAdvanced(output_dim=30)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load validation data
    X, y = load2d()
    n_train = int(len(X) * 0.8)
    X_val, y_val = X[n_train:], y[n_train:]
    
    # Visualize
    visualize_predictions(model, X_val, y_val, n_samples=args.n_samples, device=device)


def main():
    parser = argparse.ArgumentParser(
        description='Facial Keypoint Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python kfkd.py train                          # Train with default parameters
    python kfkd.py train --epochs 500 --lr 0.01  # Train with custom parameters
    python kfkd.py train_specialists              # Train specialist models
    python kfkd.py predict --model model.pth      # Generate predictions
    python kfkd.py visualize --model model.pth    # Visualize results
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Subcommands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train base model')
    train_parser.add_argument('--model', type=str, default='advanced',
                             choices=['simple', 'conv', 'dropout', 'advanced'],
                             help='Model type (default: advanced)')
    train_parser.add_argument('--epochs', type=int, default=1000,
                             help='Number of training epochs (default: 1000)')
    train_parser.add_argument('--batch-size', type=int, default=128,
                             help='Batch size (default: 128)')
    train_parser.add_argument('--lr', type=float, default=0.01,
                             help='Initial learning rate (default: 0.01)')
    train_parser.add_argument('--patience', type=int, default=100,
                             help='Early stopping patience (default: 100)')
    train_parser.add_argument('--output', type=str, default=None,
                             help='Output model path (default: model.pth)')
    train_parser.set_defaults(func=cmd_train)
    
    # Specialist model training command
    specialist_parser = subparsers.add_parser('train_specialists', 
                                              help='Train specialist models')
    specialist_parser.add_argument('--pretrain', type=str, default='model.pth',
                                   help='Pretrained model path (default: model.pth)')
    specialist_parser.add_argument('--output', type=str, default=None,
                                   help='Output path (default: specialists.pkl)')
    specialist_parser.set_defaults(func=cmd_train_specialists)
    
    # Prediction command
    predict_parser = subparsers.add_parser('predict', help='Generate predictions')
    predict_parser.add_argument('--model-path', type=str, default='model.pth',
                               help='Model path (default: model.pth)')
    predict_parser.add_argument('--output', type=str, default=None,
                               help='Output CSV path (default: predictions.csv)')
    predict_parser.set_defaults(func=cmd_predict)
    
    # Visualization command
    vis_parser = subparsers.add_parser('visualize', help='Visualize predictions')
    vis_parser.add_argument('--model-path', type=str, default='model.pth',
                           help='Model path (default: model.pth)')
    vis_parser.add_argument('--n-samples', type=int, default=6,
                           help='Number of samples to visualize (default: 6)')
    vis_parser.set_defaults(func=cmd_visualize)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
