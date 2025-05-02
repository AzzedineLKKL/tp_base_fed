import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import flwr as fl
import numpy as np
from flwr.common import (
    GetPropertiesIns, GetPropertiesRes,
    GetParametersIns, GetParametersRes,
    Parameters, FitRes, FitIns,
    EvaluateIns, EvaluateRes, Code, Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)

# Define a list of client IDs for different attack types
DATA_POISONING_CLIENT_IDS = []
MODEL_POISONING_CLIENT_IDS = [4, 6, 8]

class CustomClient(fl.client.Client):
    def __init__(
        self, 
        model: torch.nn.Module, 
        train_loader: DataLoader,
        test_loader: DataLoader, 
        device: torch.device,
        client_id: int,
        num_classes: int = 10,
        model_poisoning_factor: float = 10.0  # Factor for model poisoning
    ) -> None:
        """Initialize the Federated Learning client.
        
        Args:
            model: PyTorch model to be trained
            train_loader: DataLoader for client's training data
            test_loader: DataLoader for client's test data
            device: Device to run training on (CPU/GPU)
            client_id: Unique identifier for this client
            num_classes: Number of classes in the dataset
            model_poisoning_factor: Factor to scale model updates for model poisoning
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.client_id = client_id
        self.num_classes = num_classes
        self.model_poisoning_factor = model_poisoning_factor
        
        # Determine attack type based on client ID
        if client_id in DATA_POISONING_CLIENT_IDS:
            self.attack_type = "data"
        elif client_id in MODEL_POISONING_CLIENT_IDS:
            self.attack_type = "model"
        else:
            self.attack_type = "none"
        
        # Move model to the appropriate device
        self.model.to(self.device)
        
        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Print status
        print(f"Client {self.client_id} initialized with attack type: {self.attack_type}")
    
    def get_properties(self, instruction: GetPropertiesIns) -> GetPropertiesRes:
        """Return client properties to the server."""
        properties = {
            "dataset_size": len(self.train_loader.dataset),
            "batch_size": self.train_loader.batch_size,
            "device": str(self.device),
            "attack_type": self.attack_type,
            "client_id": self.client_id
        }
        
        return GetPropertiesRes(
            status=Status(code=Code.OK, message="Success"),
            properties=properties
        )
    
    def get_parameters(self, instruction: GetParametersIns) -> GetParametersRes:
        """Get the current model parameters."""
        model_parameters = self.model.get_model_parameters()
        parameters = ndarrays_to_parameters(model_parameters)
        
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=parameters
        )
    
    def fit(self, instruction: FitIns) -> FitRes:
        """Train the model on local data."""
        # Get parameters from server
        parameters = instruction.parameters
        config = instruction.config
        
        # Convert parameters to NumPy arrays
        model_parameters = parameters_to_ndarrays(parameters)
        
        # Update local model with server parameters
        self.model.set_model_parameters(model_parameters)
        
        # Save original parameters for model poisoning
        if self.attack_type == "model":
            original_parameters = [param.clone().detach().cpu().numpy() for param in self.model.parameters()]
        
        # Extract training configuration
        epochs = int(config.get("epochs", 1))
        
        # Train the model
        train_loss = 0.0
        train_accuracy = 0.0
        
        for _ in range(epochs):
            if self.attack_type == "data":
                # Data poisoning attack
                epoch_loss, epoch_accuracy = self._train_epoch_with_data_poisoning()
            else:
                # Normal training for both benign and model poisoning clients
                epoch_loss, epoch_accuracy = self.model.train_epoch(
                    self.train_loader, 
                    self.criterion, 
                    self.optimizer, 
                    self.device
                )
            
            train_loss = epoch_loss
            train_accuracy = epoch_accuracy
        
        # Get updated model parameters
        updated_parameters = self.model.get_model_parameters()
        
        # Apply model poisoning if needed
        if self.attack_type == "model":
            updated_parameters = self._apply_model_poisoning(updated_parameters, original_parameters)
            
            # Update the model with poisoned parameters (for consistent evaluation)
            self.model.set_model_parameters(updated_parameters)
        
        # Convert to Flower's Parameters format
        parameters_updated = ndarrays_to_parameters(updated_parameters)
        
        # Create metrics
        metrics = {
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "attack_type": self.attack_type,
            "client_id": self.client_id
        }
        
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=parameters_updated,
            num_examples=len(self.train_loader.dataset),
            metrics=metrics
        )
    
    def _train_epoch_with_data_poisoning(self) -> Tuple[float, float]:
        """Train for one epoch with data poisoning."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Apply random label poisoning
            poisoned_targets = torch.randint(0, self.num_classes, targets.shape).to(self.device)
            
            # Compute loss with poisoned labels
            loss = self.criterion(outputs, poisoned_targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Track stats (using original targets for reporting accuracy)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Calculate average loss and accuracy
        avg_loss = running_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _apply_model_poisoning(self, updated_parameters: List[np.ndarray], 
                              original_parameters: List[np.ndarray]) -> List[np.ndarray]:
        """Apply model poisoning to the trained parameters.
        
        This method modifies the model parameters after training to implement model poisoning.
        
        Args:
            updated_parameters: Parameters after training
            original_parameters: Parameters before training
            
        Returns:
            Poisoned model parameters
        """
        poisoned_parameters = []
        
        # Strategy 1: Scale the updates by a large factor to dominate aggregation
        for up, orig in zip(updated_parameters, original_parameters):
            # Calculate the update (difference between updated and original parameters)
            update = up - orig
            
            # Scale the update by a large factor
            scaled_update = update * self.model_poisoning_factor
            
            # Apply the scaled update to the original parameters
            poisoned_param = orig + scaled_update
            poisoned_parameters.append(poisoned_param)
        
        # Strategy 2: Add random noise (uncomment to use)
        # for param in updated_parameters:
        #     # Generate random noise with the same shape as the parameter
        #     noise = np.random.normal(0, 0.1, param.shape)
        #     
        #     # Add noise to parameter
        #     poisoned_param = param + noise
        #     poisoned_parameters.append(poisoned_param)
        
        # Strategy 3: Reverse the sign of updates (uncomment to use)
        # for up, orig in zip(updated_parameters, original_parameters):
        #     # Calculate update (difference between updated and original parameters)
        #     update = up - orig
        #     
        #     # Reverse the sign of the update
        #     reversed_update = -1 * update
        #     
        #     # Apply the reversed update to the original parameters
        #     poisoned_param = orig + reversed_update
        #     poisoned_parameters.append(poisoned_param)
        
        print(f"Client {self.client_id}: Applied model poisoning (scale factor: {self.model_poisoning_factor})")
        return poisoned_parameters
    
    def evaluate(self, instruction: EvaluateIns) -> EvaluateRes:
        """Evaluate the model on local test data."""
        # Get parameters from server
        parameters = instruction.parameters
        config = instruction.config
        
        # Convert parameters to NumPy arrays
        model_parameters = parameters_to_ndarrays(parameters)
        
        # Update local model with server parameters
        self.model.set_model_parameters(model_parameters)
        
        # Evaluate the model
        test_loss, test_accuracy = self.model.test_epoch(
            self.test_loader,
            self.criterion,
            self.device
        )
        
        # Create metrics
        metrics = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "attack_type": self.attack_type,
            "client_id": self.client_id
        }
        
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=float(test_loss),
            num_examples=len(self.test_loader.dataset),
            metrics=metrics
        )
    
    def to_client(self) -> 'CustomClient':
        """Return the client instance."""
        return self