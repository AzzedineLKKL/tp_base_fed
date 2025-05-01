"""Configuration parameters for the federated learning system."""

# Number of clients in the federated system
CLIENTS = 12

# Dirichlet distribution parameter (controls data heterogeneity)
# Lower alpha = more heterogeneity, higher alpha = more uniformity
ALPHA_DIRICHLET = 1

# Batch size for training and evaluation
BATCH_SIZE = 64

# Training hyperparameters
LEARNING_RATE = 0.001
NUM_EPOCHS = 3
NUM_ROUNDS = 20

# Server configuration
SERVER_ADDRESS = "[::]:8080"
MIN_CLIENTS = 2  # Minimum number of clients required to start training
CLIENT_FRACTION = 0.8  # Fraction of clients to select for each round

# Data directories
DATA_DIR = "./data"
CLIENT_DATA_DIR = "./client_datasets"