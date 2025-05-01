import subprocess
import os
import sys
import time
from config import CLIENTS  # Import the number of clients from config file

def main() -> None:
    """Launch multiple federated learning clients."""
    # Get the number of clients to run from config
    num_clients = CLIENTS
    
    print(f"Starting {num_clients} federated learning clients...")
    
    # List to keep track of client processes
    client_processes = []
    
    # Start each client as a separate process
    for cid in range(num_clients):
        # Command to run a client
        cmd = [sys.executable, "run_client.py", "--cid", str(cid)]
        
        # Start the process
        print(f"Starting client {cid}...")
        process = subprocess.Popen(cmd)
        client_processes.append(process)
        
        # Brief pause to avoid overwhelming the system
        time.sleep(0.5)
    
    print(f"\nAll {num_clients} clients have been started.")
    print("Press Ctrl+C to terminate all clients.")
    
    try:
        # Keep the script running so user can easily terminate all clients
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # When user presses Ctrl+C, terminate all client processes
        print("\nTerminating all clients...")
        for process in client_processes:
            process.terminate()
        
        # Wait for processes to terminate
        for process in client_processes:
            process.wait()
        
        print("All clients terminated.")

if __name__ == "__main__":
    main()