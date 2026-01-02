"""
Training Script
Reference: https://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

from data_loader import load2d, FacialKeypointsDataset, FLIP_INDICES
from torch.utils.data import DataLoader
from models import get_model


class EarlyStopping:
    """
    Early stopping mechanism, stops training when validation loss stops improving
    """
    
    def __init__(self, patience=100, min_delta=0, restore_best=True):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum improvement threshold
            restore_best: Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = float('inf')
        self.best_weights = None
        self.counter = 0
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
            return False


class AdjustVariable:
    """
    Dynamically adjust variables during training (e.g., learning rate, momentum)
    
    Corresponds to AdjustVariable class in the tutorial
    """
    
    def __init__(self, start, stop, n_epochs):
        """
        Args:
            start: Starting value
            stop: Ending value
            n_epochs: Total number of epochs
        """
        self.start = start
        self.stop = stop
        self.n_epochs = n_epochs
        
    def get_value(self, epoch):
        """Get value for current epoch (linear interpolation)"""
        return self.start + (self.stop - self.start) * epoch / self.n_epochs


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
    
    return total_loss / len(train_loader.dataset)


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item() * X_batch.size(0)
    
    return total_loss / len(val_loader.dataset)


def train(model, train_loader, val_loader, n_epochs=1000, lr_start=0.01, lr_end=0.001,
          momentum_start=0.9, momentum_end=0.999, patience=100, device='cuda'):
    """
    Train the model
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        n_epochs: Number of training epochs
        lr_start: Initial learning rate
        lr_end: Final learning rate
        momentum_start: Initial momentum
        momentum_end: Final momentum
        patience: Early stopping patience
        device: Computing device
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    
    # Use SGD optimizer with momentum
    optimizer = optim.SGD(model.parameters(), lr=lr_start, momentum=momentum_start, 
                         nesterov=True)
    
    # Dynamically adjust learning rate and momentum
    lr_adjuster = AdjustVariable(lr_start, lr_end, n_epochs)
    momentum_adjuster = AdjustVariable(momentum_start, momentum_end, n_epochs)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, restore_best=True)
    
    # Record training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'momentum': []
    }
    
    print(f"Training on {device}...")
    print(f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")
    print("-" * 60)
    
    for epoch in range(n_epochs):
        # Update learning rate and momentum
        lr = lr_adjuster.get_value(epoch)
        momentum = momentum_adjuster.get_value(epoch)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            param_group['momentum'] = momentum
        
        # Train and validate
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(lr)
        history['momentum'].append(momentum)
        
        # Calculate RMSE (convert back to original coordinate space)
        train_rmse = np.sqrt(train_loss) * 48
        val_rmse = np.sqrt(val_loss) * 48
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{n_epochs} | "
                  f"Train RMSE: {train_rmse:.4f} | "
                  f"Val RMSE: {val_rmse:.4f} | "
                  f"LR: {lr:.6f}")
        
        # Check early stopping
        if early_stopping(val_loss, model):
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    return history


def plot_learning_curves(history, save_path=None):
    """Plot learning curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Convert to RMSE
    train_rmse = [np.sqrt(l) * 48 for l in history['train_loss']]
    val_rmse = [np.sqrt(l) * 48 for l in history['val_loss']]
    
    axes[0].plot(epochs, train_rmse, 'b--', label='Train RMSE', alpha=0.7)
    axes[0].plot(epochs, val_rmse, 'r-', label='Validation RMSE', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('RMSE (pixels)')
    axes[0].set_title('Learning Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate and momentum
    ax1 = axes[1]
    ax2 = ax1.twinx()
    
    line1, = ax1.plot(epochs, history['lr'], 'b-', label='Learning Rate')
    line2, = ax2.plot(epochs, history['momentum'], 'r-', label='Momentum')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Learning Rate', color='b')
    ax2.set_ylabel('Momentum', color='r')
    ax1.set_title('Learning Rate and Momentum Schedule')
    ax1.legend(handles=[line1, line2], loc='center right')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Learning curves saved to: {save_path}")
    
    plt.show()


def visualize_predictions(model, X, y, n_samples=6, device='cuda'):
    """Visualize model predictions"""
    model.eval()
    
    # Randomly select samples
    indices = np.random.choice(len(X), n_samples, replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img = X[idx].reshape(96, 96)
            
            # Get predictions
            X_tensor = torch.FloatTensor(X[idx]).unsqueeze(0).to(device)
            pred = model(X_tensor).cpu().numpy().flatten()
            
            # Ground truth
            true = y[idx]
            
            # Restore coordinates
            pred_coords = pred * 48 + 48
            true_coords = true * 48 + 48
            
            # Plot image
            axes[i].imshow(img, cmap='gray')
            
            # Plot ground truth keypoints (green)
            axes[i].scatter(true_coords[0::2], true_coords[1::2], 
                          c='green', marker='+', s=100, label='Ground Truth')
            
            # Plot predicted keypoints (red)
            axes[i].scatter(pred_coords[0::2], pred_coords[1::2], 
                          c='red', marker='x', s=100, label='Prediction')
            
            axes[i].set_title(f'Sample {idx}')
            axes[i].axis('off')
            
            if i == 0:
                axes[i].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
    print("Prediction visualization saved to: predictions.png")
    plt.show()


def main():
    """Main training function"""
    # Set device
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
    
    # Create data loaders (with data augmentation)
    train_dataset = FacialKeypointsDataset(X_train, y_train, augment=True, 
                                           flip_indices=FLIP_INDICES)
    val_dataset = FacialKeypointsDataset(X_val, y_val, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False,
                           num_workers=0, pin_memory=True)
    
    # Create model
    print("\nCreating model...")
    model = get_model('advanced', output_dim=30)
    
    # Print model parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Train model
    print("\nStarting training...")
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=1000,
        lr_start=0.01,
        lr_end=0.001,
        momentum_start=0.9,
        momentum_end=0.999,
        patience=100,
        device=device
    )
    
    # Plot learning curves
    print("\nPlotting learning curves...")
    plot_learning_curves(history, 'learning_curves.png')
    
    # Visualize predictions
    print("\nVisualizing predictions...")
    visualize_predictions(model, X_val, y_val, device=device)
    
    # Save model
    print("\nSaving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
    }, 'model.pth')
    print("Model saved to: model.pth")
    
    # Final results
    final_val_rmse = np.sqrt(history['val_loss'][-1]) * 48
    print(f"\nFinal validation RMSE: {final_val_rmse:.4f} pixels")


if __name__ == "__main__":
    main()
