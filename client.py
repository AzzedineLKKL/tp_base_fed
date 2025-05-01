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

class CustomClient(fl.client.Client):
    def __init__(
        self, 
        model: torch.nn.Module, 
        train_loader: DataLoader,
        test_loader: DataLoader, 
        device: torch.device
    ) -> None:
        """Initialize the Federated Learning client.
        
        Args:
            model: PyTorch model to be trained
            train_loader: DataLoader for client's training data
            test_loader: DataLoader for client's test data
            device: Device to run training on (CPU/GPU)
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # Move model to the appropriate device
        self.model.to(self.device)
        
        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
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
            "device": str(self.device)
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
            "train_accuracy": float(train_accuracy)
        }
        
        # Create and return response
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=parameters_updated,
            num_examples=len(self.train_loader.dataset),
            metrics=metrics
        )
    
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
            "test_accuracy": float(test_accuracy)
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