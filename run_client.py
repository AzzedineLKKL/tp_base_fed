import argparse
import torch
import flwr as fl
from typing import Tuple
from torch.utils.data import DataLoader

# Import necessary functions and classes
from main2 import load_client_data
from model import CustomFashionModel
from client import CustomClient
from config import BATCH_SIZE

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument(
        "--cid", 
        type=int, 
        required=True, 
        help="Client ID (required)"
    )
    parser.add_argument(
        "--server_address", 
        type=str, 
        default="localhost:8080", 
        help="Server address (default: [::]:8080)"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="./client_datasets", 
        help="Directory containing client data (default: ./client_datasets)"
    )
    
    return parser.parse_args()

def load_data(cid: int, data_dir: str) -> Tuple[DataLoader, DataLoader]:
    """Load client data using the previously implemented function."""
    return load_client_data(cid=cid, data_dir=data_dir, batch_size=BATCH_SIZE)

def main():
    """Run the federated learning client."""
    # Parse command-line arguments
    args = parse_args()
    
    # Set device for training (use GPU if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load client data
    print(f"Loading data for client {args.cid}...")
    try:
        train_loader, val_loader = load_data(cid=args.cid, data_dir=args.data_dir)
        print(f"Loaded {len(train_loader.dataset)} training samples and {len(val_loader.dataset)} validation samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure you've generated client datasets using:")
        print("python -c \"from data_utils import generate_distributed_datasets; from config import CLIENTS, ALPHA_DIRICHLET; generate_distributed_datasets(CLIENTS, ALPHA_DIRICHLET, './client_datasets')\"")
        return
    
    # Instantiate the model
    model = CustomFashionModel()
    
    # Create the client
    client = CustomClient(
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,
        client_id=args.cid,
        device=device
    )
    
    # Start the client and connect to the server
    print(f"Starting client {args.cid}...")
    print(f"Connecting to server at {args.server_address}")
    try:
        fl.client.start_client(server_address=args.server_address, client=client.to_client())
        print(f"Client {args.cid} finished")
    except Exception as e:
        print(f"Error connecting to server: {e}")
        print("Make sure the server is running before starting clients.")

if __name__ == "__main__":
    main()