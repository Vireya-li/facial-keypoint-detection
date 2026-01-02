"""
Neural Network Model Definitions
Reference: https://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    """
    Simple single hidden layer neural network (corresponds to net1 in the tutorial)
    
    Architecture:
        Input (9216) -> Hidden (100) -> Output (30)
    """
    
    def __init__(self, output_dim=30):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(9216, 100)
        self.fc2 = nn.Linear(100, output_dim)
        
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvNet(nn.Module):
    """
    Convolutional Neural Network (corresponds to net2 in the tutorial)
    
    Architecture:
        Conv1 (1->32, 3x3) -> MaxPool (2x2) ->
        Conv2 (32->64, 2x2) -> MaxPool (2x2) ->
        Conv3 (64->128, 2x2) -> MaxPool (2x2) ->
        FC1 (1000) -> FC2 (1000) -> Output (30)
    """
    
    def __init__(self, output_dim=30):
        super(ConvNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, padding=0)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate flattened feature dimensions
        # Input: 96x96
        # After conv1 (3x3) -> 94x94 -> pool -> 47x47
        # After conv2 (2x2) -> 46x46 -> pool -> 23x23
        # After conv3 (2x2) -> 22x22 -> pool -> 11x11
        # 11 * 11 * 128 = 15488
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 11 * 11, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, output_dim)
        
    def forward(self, x):
        # Convolutional layers + activation + pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class ConvNetDropout(nn.Module):
    """
    Convolutional Neural Network with Dropout (corresponds to net4/net5 in the tutorial)
    
    Dropout helps prevent overfitting
    """
    
    def __init__(self, output_dim=30, dropout1=0.1, dropout2=0.2, 
                 dropout3=0.3, dropout4=0.5, dropout5=0.5):
        super(ConvNetDropout, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        self.dropout3 = nn.Dropout(dropout3)
        self.dropout4 = nn.Dropout(dropout4)
        self.dropout5 = nn.Dropout(dropout5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 11 * 11, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, output_dim)
        
    def forward(self, x):
        # Convolutional layers + activation + pooling + dropout
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers + dropout
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout5(x)
        
        x = self.fc3(x)
        
        return x


class ConvNetAdvanced(nn.Module):
    """
    Advanced Convolutional Neural Network (corresponds to net6/net7 in the tutorial)
    
    Features:
        - More convolutional layers
        - Larger convolution kernels
        - Dropout regularization
    """
    
    def __init__(self, output_dim=30, dropout_conv=0.1, dropout_fc=0.5):
        super(ConvNetAdvanced, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv1_bn = nn.BatchNorm2d(32)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        
        # Fourth convolutional block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(256)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout_conv = nn.Dropout2d(dropout_conv)
        self.dropout_fc = nn.Dropout(dropout_fc)
        
        # Calculate flattened size: 96 -> 48 -> 24 -> 12 -> 6
        # 6 * 6 * 256 = 9216
        self.fc1 = nn.Linear(256 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_dim)
        
    def forward(self, x):
        # Convolutional block 1
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.dropout_conv(x)
        
        # Convolutional block 2
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.dropout_conv(x)
        
        # Convolutional block 3
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.dropout_conv(x)
        
        # Convolutional block 4
        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))
        x = self.dropout_conv(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)
        
        x = self.fc3(x)
        
        return x


def get_model(model_type='advanced', output_dim=30, **kwargs):
    """
    Get the specified model type
    
    Args:
        model_type: Model type ('simple', 'conv', 'dropout', 'advanced')
        output_dim: Output dimension
        **kwargs: Other model parameters
        
    Returns:
        model: Neural network model
    """
    models = {
        'simple': SimpleNet,
        'conv': ConvNet,
        'dropout': ConvNetDropout,
        'advanced': ConvNetAdvanced,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type](output_dim=output_dim, **kwargs)


if __name__ == "__main__":
    # Test models
    print("Testing models...")
    
    # Create random input
    x = torch.randn(2, 1, 96, 96)
    
    for model_type in ['simple', 'conv', 'dropout', 'advanced']:
        model = get_model(model_type)
        
        if model_type == 'simple':
            out = model(x.view(2, -1))
        else:
            out = model(x)
            
        print(f"{model_type}: output shape = {out.shape}")
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Number of parameters: {n_params:,}")
