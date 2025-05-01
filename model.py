import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Tuple

class CustomFashionModel(nn.Module):
    def __init__(self) -> None:
        """Initialize the model architecture for Fashion MNIST classification."""
        super(CustomFashionModel, self).__init__()
        
        # Simple CNN architecture
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes in Fashion MNIST
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def train_epoch(self, train_loader: DataLoader, 
                    criterion: nn.Module, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> Tuple[float, float]:
        """Train the model for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            criterion: Loss function
            optimizer: Optimizer for model parameters
            device: Device to run training on (CPU/GPU)
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track stats
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Calculate average loss and accuracy
        avg_loss = running_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def test_epoch(self, test_loader: DataLoader,
                  criterion: nn.Module,
                  device: torch.device) -> Tuple[float, float]:
        """Evaluate the model on test data.
        
        Args:
            test_loader: DataLoader for test data
            criterion: Loss function
            device: Device to run evaluation on (CPU/GPU)
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.eval()  # Set model to evaluation mode
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():  # No gradients needed for evaluation
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                
                # Track stats
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calculate average loss and accuracy
        avg_loss = running_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def get_model_parameters(self) -> List[np.ndarray]:
        """Extract model parameters as a list of NumPy arrays.
        
        Returns:
            List of model parameters as NumPy arrays
        """
        return [param.cpu().detach().numpy() for param in self.parameters()]
    
    def set_model_parameters(self, parameters: List[np.ndarray]) -> None:
        """Update model parameters from a list of NumPy arrays.
        
        Args:
            parameters: List of model parameters as NumPy arrays
        """
        for param, param_data in zip(self.parameters(), parameters):
            param.data = torch.from_numpy(param_data).to(param.device)