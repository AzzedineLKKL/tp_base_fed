import argparse
import time
import json
import os
from typing import Dict, List, Tuple, Optional

import flwr as fl
from flwr.common import Parameters
from flwr.server import ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History

from server import CustomClientManager, FedAvgStrategy
from model import CustomFashionModel
from config import MIN_CLIENTS, CLIENT_FRACTION, NUM_ROUNDS, NUM_EPOCHS

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument(
        "--min_clients", 
        type=int, 
        default=MIN_CLIENTS, 
        help=f"Minimum number of clients (default: {MIN_CLIENTS})"
    )
    parser.add_argument(
        "--fraction", 
        type=float, 
        default=CLIENT_FRACTION, 
        help=f"Fraction of clients to use for training (default: {CLIENT_FRACTION})"
    )
    parser.add_argument(
        "--num_rounds", 
        type=int, 
        default=NUM_ROUNDS, 
        help=f"Number of training rounds (default: {NUM_ROUNDS})"
    )
    parser.add_argument(
        "--server_address", 
        type=str, 
        default="[::]:8080", 
        help="Server address (default: [::]:8080)"
    )
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default="./results", 
        help="Directory to save results (default: ./results)"
    )
    
    return parser.parse_args()

def fit_config(server_round: int) -> Dict[str, str]:
    """Return training configuration for clients."""
    return {
        "epochs": str(NUM_EPOCHS),
        "server_round": str(server_round),
    }

def evaluate_config(server_round: int) -> Dict[str, str]:
    """Return evaluation configuration for clients."""
    return {
        "server_round": str(server_round),
    }

# Create a custom client manager that prints when clients join
class PrintingClientManager(CustomClientManager):
    """Custom client manager that prints when clients join."""
    
    def register(self, client: ClientProxy) -> bool:
        """Register a client and print a message."""
        success = super().register(client)
        if success:
            print(f"\n🔵 New client joined! Client ID: {client.cid}")
            print(f"Total clients connected: {self.num_available()}")
        return success

# Create a waiting strategy that won't proceed until enough clients are available
class WaitingFedAvgStrategy(FedAvgStrategy):
    """Strategy that waits for minimum clients before starting."""
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: CustomClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """Wait for minimum clients before configuring training."""
        # Keep checking for enough clients
        while client_manager.num_available() < self.min_available_clients:
            print(f"\rWaiting for clients: {client_manager.num_available()}/{self.min_available_clients} connected... Press Ctrl+C to exit", end="")
            time.sleep(2)
            
        # If we get here, we have enough clients
        if server_round == 1:  # Only print this message once at the start
            print(f"\n\n✅ {client_manager.num_available()} clients connected - minimum requirement met!")
            print(f"Starting federated learning with {self.min_fit_clients} clients per round...")
            
        # Continue with normal configuration
        return super().configure_fit(server_round, parameters, client_manager)
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: CustomClientManager
    ) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        """Wait for minimum clients before configuring evaluation."""
        # Keep checking for enough clients (if somehow clients left during training)
        if client_manager.num_available() < self.min_available_clients:
            print(f"\nNot enough clients for evaluation. Waiting for {self.min_available_clients} clients...")
            while client_manager.num_available() < self.min_available_clients:
                print(f"\rWaiting for clients: {client_manager.num_available()}/{self.min_available_clients} connected...", end="")
                time.sleep(2)
            print(f"\n✅ {client_manager.num_available()} clients connected - continuing with evaluation...")
            
        # Continue with normal configuration
        return super().configure_evaluate(server_round, parameters, client_manager)
    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, float]]]:
        """Implement required evaluate method."""
        if self.evaluate_fn is None:
            return None
        
        weights = fl.common.parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, weights, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

def save_results(history: History, results_dir: str, experiment_name: str = None) -> str:
    """Save the training results to a JSON file.
    
    Args:
        history: Flower server history object
        results_dir: Directory to save results
        experiment_name: Optional experiment name for the filename
        
    Returns:
        Path to the saved results file
    """
    # Create directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if experiment_name:
        filename = f"{experiment_name}_{timestamp}.json"
    else:
        filename = f"federated_results_{timestamp}.json"
    
    file_path = os.path.join(results_dir, filename)
    
    # Extract results from history
    results = {
        "loss": {
            "distributed": history.losses_distributed
        },
        "metrics": {
            "distributed": {
                "fit": {},
                "evaluate": {}
            }
        }
    }
    
    # Add training metrics
    for metric_name in history.metrics_distributed_fit:
        results["metrics"]["distributed"]["fit"][metric_name] = [
            [round_num, value] for round_num, value in history.metrics_distributed_fit[metric_name]
        ]
    
    # Add evaluation metrics
    for metric_name in history.metrics_distributed:
        results["metrics"]["distributed"]["evaluate"][metric_name] = [
            [round_num, value] for round_num, value in history.metrics_distributed[metric_name]
        ]
    
    # Save to JSON file
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n📊 Results saved to: {file_path}")
    return file_path

def main():
    """Run the federated learning server."""
    # Parse command-line arguments
    args = parse_args()
    
    # Create custom client manager with printing capabilities
    client_manager = PrintingClientManager()
    
    # Initialize model parameters (if needed)
    model = CustomFashionModel()
    initial_parameters = fl.common.ndarrays_to_parameters(model.get_model_parameters())
    
    # Create strategy with waiting behavior
    strategy = WaitingFedAvgStrategy(
        fraction_fit=args.fraction,
        fraction_evaluate=args.fraction,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=initial_parameters,
    )
    
    # Print instructions
    print(f"Starting server at {args.server_address}")
    print(f"The server will wait for {args.min_clients} clients before starting training.")
    print("\nIMPORTANT: Start client processes in separate terminals using:")
    print(f"  python run_client.py --cid 0")
    print(f"  python run_client.py --cid 1")
    print("  ... and so on for additional clients")
    print("\nThe server will wait indefinitely until enough clients connect.")
    
    # Create server with the custom client manager
    server = fl.server.Server(client_manager=client_manager, strategy=strategy)
    
    # Start Flower server
    history = fl.server.start_server(
        server_address=args.server_address,
        server=server,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
    )
    
    # Save results after training completes
    if history:
        experiment_name = f"federated_fashion_mnist_{args.min_clients}clients_{args.num_rounds}rounds"
        results_file = save_results(history, args.results_dir, experiment_name)
        
        print("\n🎉 Training complete! You can analyze the results with:")
        print(f"  python analyze_results.py {results_file}")

if __name__ == "__main__":
    main()