import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import flwr as fl
from flwr.common import (
    GetPropertiesIns, GetPropertiesRes,
    GetParametersIns, GetParametersRes,
    Parameters, FitRes, FitIns,
    EvaluateIns, EvaluateRes, Code, Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)

# Define a list of attacker client IDs
ATTACKER_CLIENT_IDS = [2, 5, 7,4,6]  # Clients with these IDs will be attackers

class CustomClient(fl.client.Client):
    def __init__(
        self, 
        model: torch.nn.Module, 
        train_loader: DataLoader,
        test_loader: DataLoader, 
        device: torch.device,
        client_id: int,  # Add client_id parameter
        num_classes: int = 10
    ) -> None:
        """Initialize the Federated Learning client.
        
        Args:
            model: PyTorch model to be trained
            train_loader: DataLoader for client's training data
            test_loader: DataLoader for client's test data
            device: Device to run training on (CPU/GPU)
            client_id: Unique identifier for this client
            num_classes: Number of classes in the dataset
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.client_id = client_id
        self.is_attacker = client_id in ATTACKER_CLIENT_IDS  # Determine if attacker based on ID
        self.num_classes = num_classes
        
        # Move model to the appropriate device
        self.model.to(self.device)
        
        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Print status
        status = "ATTACKER" if self.is_attacker else "regular client"
        print(f"Client {self.client_id} initialized as {status}")
    
    def get_properties(self, instruction: GetPropertiesIns) -> GetPropertiesRes:
        """Return client properties to the server.
        
        Args:
            instruction: Server's request for properties
            
        Returns:
            Client properties response
        """
        # Example properties - customize as needed
        properties = {
            "dataset_size": len(self.train_loader.dataset),
            "batch_size": self.train_loader.batch_size,
            "device": str(self.device),
            "is_attacker": self.is_attacker,  # Include attacker status in properties
            "client_id": self.client_id
        }
        
        # Create and return response
        return GetPropertiesRes(
            status=Status(code=Code.OK, message="Success"),
            properties=properties
        )
    
    def get_parameters(self, instruction: GetParametersIns) -> GetParametersRes:
        """Get the current model parameters.
        
        Args:
            instruction: Server's request for parameters
            
        Returns:
            Model parameters response
        """
        # Extract model parameters as NumPy arrays
        model_parameters = self.model.get_model_parameters()
        
        # Convert to Flower's Parameters format
        parameters = ndarrays_to_parameters(model_parameters)
        
        # Create and return response
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=parameters
        )
    
    def fit(self, instruction: FitIns) -> FitRes:
        """Train the model on local data.
        
        Args:
            instruction: Server's fit instructions containing model parameters and config
            
        Returns:
            Training results
        """
        # Get parameters from server
        parameters = instruction.parameters
        config = instruction.config
        
        # Convert parameters to NumPy arrays
        model_parameters = parameters_to_ndarrays(parameters)
        
        # Update local model with server parameters
        self.model.set_model_parameters(model_parameters)
        
        # Extract training configuration
        epochs = int(config.get("epochs", 1))
        
        # Train the model
        train_loss = 0.0
        train_accuracy = 0.0
        
        for _ in range(epochs):
            if self.is_attacker:
                # Attacker client - use poisoned training
                epoch_loss, epoch_accuracy = self._train_epoch_with_poisoning()
            else:
                # Normal client - use standard training
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
        
        # Convert to Flower's Parameters format
        parameters_updated = ndarrays_to_parameters(updated_parameters)
        
        # Create metrics
        metrics = {
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "is_attacker": self.is_attacker,
            "client_id": self.client_id
        }
        
        # Create and return response
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=parameters_updated,
            num_examples=len(self.train_loader.dataset),
            metrics=metrics
        )
    
    def _train_epoch_with_poisoning(self) -> Tuple[float, float]:
        """Train for one epoch with data poisoning.
        
        This method modifies labels before computing loss to simulate a data poisoning attack.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
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
            
            # Apply label flipping attack - various strategies can be used:
            
            # Strategy 1: Flip all labels (e.g., for a 10-class problem, label y becomes (9-y))
            poisoned_targets = torch.randint(0, self.num_classes, targets.shape).to(self.device)
            
            # Strategy 2: Random label assignment (uncomment to use)
            # poisoned_targets = torch.randint(0, self.num_classes, targets.shape).to(self.device)
            
            # Strategy 3: Target class attack (e.g., change all labels to class 0)
            # poisoned_targets = torch.zeros_like(targets)
            
            # Compute loss with poisoned labels
            loss = self.criterion(outputs, poisoned_targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Track stats (using original targets for reporting accuracy)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()  # Compare with original targets
        
        # Calculate average loss and accuracy
        avg_loss = running_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, instruction: EvaluateIns) -> EvaluateRes:
        """Evaluate the model on local test data.
        
        Args:
            instruction: Server's evaluation instructions
            
        Returns:
            Evaluation results
        """
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
            "is_attacker": self.is_attacker,
            "client_id": self.client_id
        }
        
        # Create and return response
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=float(test_loss),
            num_examples=len(self.test_loader.dataset),
            metrics=metrics
        )
    
    def to_client(self) -> 'CustomClient':
        """Return the client instance.
        
        Returns:
            The client itself
        """
        return self