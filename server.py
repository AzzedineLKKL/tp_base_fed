import random
import threading
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from flwr.common import (
    Parameters,
    Scalar,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from model import CustomFashionModel  # Import the model from Step 2


class CustomClientManager(ClientManager):
    """Custom client manager for federated learning server."""
    
    def __init__(self):
        """Initialize the client manager with an empty client dictionary and a lock."""
        self.clients: Dict[str, ClientProxy] = {}
        self.lock = threading.RLock()  # Use RLock for reentrant locking
        
    def num_available(self) -> int:
        """Return the number of available clients.
        
        Returns:
            Number of available clients
        """
        with self.lock:
            return len(self.clients)
    
    def register(self, client: ClientProxy) -> bool:
        """Register a new client.
        
        Args:
            client: Client to register
            
        Returns:
            True if registration was successful, False otherwise
        """
        with self.lock:
            if client.cid in self.clients:
                return False  # Client already registered
            self.clients[client.cid] = client
            return True
    
    def unregister(self, client: ClientProxy) -> None:
        """Unregister a client.
        
        Args:
            client: Client to unregister
        """
        with self.lock:
            if client.cid in self.clients:
                del self.clients[client.cid]
    
    def all(self) -> Dict[str, ClientProxy]:
        """Return all registered clients.
        
        Returns:
            Dictionary of all registered clients
        """
        with self.lock:
            return self.clients.copy()  # Return a copy to prevent race conditions
    
    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        """Wait for a specified number of clients to become available.
        
        Args:
            num_clients: Minimum number of clients to wait for
            timeout: Maximum time to wait in seconds (default: 1 day)
            
        Returns:
            True if the required number of clients is available, False otherwise
        """
        if num_clients < 1:
            return True
        
        # Create an event to wait on
        available = threading.Event()
        
        # Function to set the event when enough clients are available
        def check_clients():
            if self.num_available() >= num_clients:
                available.set()
        
        # Set up a periodic check
        def periodic_check():
            check_clients()
            if not available.is_set() and timeout > 0:
                # Schedule another check in 1 second
                threading.Timer(1.0, periodic_check).start()
        
        # Start checking
        periodic_check()
        
        # Wait for the event or timeout
        return available.wait(timeout=timeout)
    
    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[object] = None,
    ) -> List[ClientProxy]:
        """Sample a subset of available clients.
        
        Args:
            num_clients: Number of clients to sample
            min_num_clients: Minimum number of clients to sample
            criterion: Optional selection criterion
            
        Returns:
            List of sampled clients
        """
        if min_num_clients is None:
            min_num_clients = num_clients
        
        with self.lock:
            # Get all clients
            all_clients = list(self.clients.values())
            
            # Check if we have enough clients
            if len(all_clients) < min_num_clients:
                return []
            
            # Sample clients randomly
            sampled_clients = random.sample(
                all_clients, 
                min(num_clients, len(all_clients))
            )
            
            return sampled_clients


class FedAvgStrategy(Strategy):
    """Federated Averaging strategy implementation."""
   
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures: bool = True,
        initial_parameters=None,
    ):
        """Initialize the FedAvg strategy.
        
        Args:
            fraction_fit: Fraction of clients to sample for training
            fraction_evaluate: Fraction of clients to sample for evaluation
            min_fit_clients: Minimum number of clients for training
            min_evaluate_clients: Minimum number of clients for evaluation
            min_available_clients: Minimum number of available clients required
            evaluate_fn: Optional function for centralized evaluation
            on_fit_config_fn: Function to generate client fit configuration
            on_evaluate_config_fn: Function to generate client evaluate configuration
            accept_failures: Whether to accept client failures
            initial_parameters: Initial global model parameters
        """
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
    
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters.
        
        Args:
            client_manager: Client manager instance
            
        Returns:
            Initial parameters or None
        """
        if self.initial_parameters is not None:
            return self.initial_parameters
        
        # If initial parameters not provided, create a new model and get parameters
        model = CustomFashionModel()
        ndarrays = model.get_model_parameters()
        return ndarrays_to_parameters(ndarrays)
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    
        if self.evaluate_fn is None:
        # No central evaluation function provided
            return None
    
    # If an evaluation function was provided, use it
        ndarrays = parameters_to_ndarrays(parameters)
        loss, metrics = self.evaluate_fn(server_round, ndarrays, {})
        return loss, metrics
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.
        
        Args:
            server_round: Current server round
            parameters: Global model parameters
            client_manager: Client manager instance
            
        Returns:
            List of (client, fit_instructions) tuples
        """
        # Sample clients
        sample_size = int(client_manager.num_available() * self.fraction_fit)
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_fit_clients,
        )
        
        # Create fit instructions
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        fit_instructions = []
        for client in clients:
            fit_ins = FitIns(parameters=parameters, config=config)
            fit_instructions.append((client, fit_ins))
        
        return fit_instructions
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client training results using FedAvg.
        
        Args:
            server_round: Current server round
            results: List of (client, fit_result) tuples
            failures: List of failed client operations
            
        Returns:
            Tuple of (aggregated_parameters, metrics)
        """
        # Handle failures
        if not results:
            return None, {}
        
        # Convert results to (weights, num_examples) tuples
        weights_results = []
        
        for _, fit_res in results:
            parameters = fit_res.parameters
            num_examples = fit_res.num_examples
            
            # Convert Parameters to NumPy arrays
            ndarrays = parameters_to_ndarrays(parameters)
            weights_results.append((ndarrays, num_examples))
        
        # Aggregate parameters using weighted average
        aggregated_ndarrays = self.aggregate_parameters(weights_results)
        
        # Aggregate metrics
        metrics_aggregated = {}
        if len(results) > 0:
            metrics_aggregated["train_loss"] = np.mean([
                res.metrics["train_loss"] for _, res in results
            ])
            metrics_aggregated["train_accuracy"] = np.mean([
                res.metrics["train_accuracy"] for _, res in results
            ])
        
        return ndarrays_to_parameters(aggregated_ndarrays), metrics_aggregated
    
    def aggregate_parameters(self, weights_results):
        """Aggregate model parameters using FedAvg.
        
        Args:
            weights_results: List of (weights, num_examples) tuples
            
        Returns:
            Aggregated model parameters
        """
        # Get total number of examples
        total_examples = sum(num_examples for _, num_examples in weights_results)
        
        # Create a list to hold the weighted parameters
        weighted_params = []
        
        # For each set of client parameters
        for idx, (params, num_examples) in enumerate(weights_results):
            # Calculate weight based on number of examples
            weight = num_examples / total_examples
            
            if idx == 0:
                # Initialize weighted_params with the first client's weighted parameters
                weighted_params = [weight * param for param in params]
            else:
                # Add the weighted parameters of the current client
                for i, param in enumerate(params):
                    weighted_params[i] += weight * param
        
        return weighted_params
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the evaluation process for clients.
        
        Args:
            server_round: Current server round
            parameters: Global model parameters
            client_manager: Client manager instance
            
        Returns:
            List of (client, evaluation_instructions) tuples
        """
        # Sample clients for evaluation
        sample_size = int(client_manager.num_available() * self.fraction_evaluate)
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_evaluate_clients,
        )
        
        # Create evaluation instructions
        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        
        evaluation_instructions = []
        for client in clients:
            evaluate_ins = EvaluateIns(parameters=parameters, config=config)
            evaluation_instructions.append((client, evaluate_ins))
        
        return evaluation_instructions
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results from clients.
        
        Args:
            server_round: Current server round
            results: List of (client, evaluate_result) tuples
            failures: List of failed client operations
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Handle failures
        if not results:
            return None, {}
        
        # Aggregate loss and metrics
        total_loss = 0.0
        total_examples = 0
        
        # Aggregate metrics dictionary
        metrics_aggregated = {}
        
        for _, evaluate_res in results:
            loss = evaluate_res.loss
            num_examples = evaluate_res.num_examples
            
            # Weighted loss
            total_loss += loss * num_examples
            total_examples += num_examples
            
            # Aggregate metrics from each client
            if evaluate_res.metrics:
                for key, value in evaluate_res.metrics.items():
                    if key not in metrics_aggregated:
                        metrics_aggregated[key] = 0.0
                    metrics_aggregated[key] += value * num_examples
        
        # Calculate average loss
        average_loss = total_loss / total_examples if total_examples > 0 else None
        
        # Calculate average metrics
        for key in metrics_aggregated:
            metrics_aggregated[key] /= total_examples
        
        return average_loss, metrics_aggregated