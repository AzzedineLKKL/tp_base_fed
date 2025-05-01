import json
import os
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from typing import Dict, List, Any

class ResultsVisualizer:
    def __init__(self) -> None:
        """Initialize the results visualizer."""
        self.results = None
        self.metric_names = []
        self.rounds = 0
    
    def load_simulation_results(self, file_name: str) -> None:
        """Load the JSON file containing the simulation results.
        
        Args:
            file_name: Path to the JSON file
        """
        try:
            with open(file_name, 'r') as f:
                self.results = json.load(f)
            
            # Extract metric names and number of rounds
            if self.results:
                # Get all available metrics
                self.metric_names = []
                if 'metrics' in self.results:
                    if 'distributed' in self.results['metrics']:
                        if 'fit' in self.results['metrics']['distributed']:
                            self.metric_names.extend(list(self.results['metrics']['distributed']['fit'].keys()))
                        if 'evaluate' in self.results['metrics']['distributed']:
                            self.metric_names.extend(list(self.results['metrics']['distributed']['evaluate'].keys()))
                
                # Get number of rounds
                if 'loss' in self.results and 'distributed' in self.results['loss']:
                    self.rounds = len(self.results['loss']['distributed'])
                
                print(f"Successfully loaded results with {self.rounds} rounds and metrics: {', '.join(self.metric_names)}")
            else:
                print("Warning: Loaded file contains no results")
        except Exception as e:
            print(f"Error loading results: {e}")
    
    def plot_results(self, fig_directory: str) -> None:
        """Plot all evaluation metrics over training rounds.
        
        Args:
            fig_directory: Directory to save the figures
        """
        if not self.results:
            print("No results loaded. Call load_simulation_results first.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(fig_directory, exist_ok=True)
        
        # Plot loss
        if 'loss' in self.results and 'distributed' in self.results['loss']:
            plt.figure(figsize=(10, 6))
            rounds = list(range(1, self.rounds + 1))
            losses = self.results['loss']['distributed']
            
            plt.plot(rounds, losses, 'o-', linewidth=2, markersize=8)
            plt.title('Federated Learning Loss over Rounds', fontsize=16)
            plt.xlabel('Round', fontsize=14)
            plt.ylabel('Loss', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xticks(rounds)
            plt.tight_layout()
            
            # Save the figure
            loss_path = os.path.join(fig_directory, 'loss.png')
            plt.savefig(loss_path)
            plt.close()
            print(f"Loss plot saved to {loss_path}")
        
        # Plot training metrics
        if 'metrics' in self.results and 'distributed' in self.results['metrics']:
            if 'fit' in self.results['metrics']['distributed']:
                train_metrics = self.results['metrics']['distributed']['fit']
                for metric_name, metric_values in train_metrics.items():
                    plt.figure(figsize=(10, 6))
                    
                    # Extract rounds and values from the data
                    rounds = [item[0] for item in metric_values]
                    values = [item[1] for item in metric_values]
                    
                    plt.plot(rounds, values, 'o-', linewidth=2, markersize=8, color='blue')
                    plt.title(f'Training {metric_name} over Rounds', fontsize=16)
                    plt.xlabel('Round', fontsize=14)
                    plt.ylabel(metric_name, fontsize=14)
                    plt.grid(True, alpha=0.3)
                    plt.xticks(rounds)
                    plt.tight_layout()
                    
                    # Save the figure
                    metric_path = os.path.join(fig_directory, f'train_{metric_name}.png')
                    plt.savefig(metric_path)
                    plt.close()
                    print(f"Training {metric_name} plot saved to {metric_path}")
            
            # Plot evaluation metrics
            if 'evaluate' in self.results['metrics']['distributed']:
                eval_metrics = self.results['metrics']['distributed']['evaluate']
                for metric_name, metric_values in eval_metrics.items():
                    plt.figure(figsize=(10, 6))
                    
                    # Extract rounds and values from the data
                    rounds = [item[0] for item in metric_values]
                    values = [item[1] for item in metric_values]
                    
                    plt.plot(rounds, values, 'o-', linewidth=2, markersize=8, color='green')
                    plt.title(f'Evaluation {metric_name} over Rounds', fontsize=16)
                    plt.xlabel('Round', fontsize=14)
                    plt.ylabel(metric_name, fontsize=14)
                    plt.grid(True, alpha=0.3)
                    plt.xticks(rounds)
                    plt.tight_layout()
                    
                    # Save the figure
                    metric_path = os.path.join(fig_directory, f'eval_{metric_name}.png')
                    plt.savefig(metric_path)
                    plt.close()
                    print(f"Evaluation {metric_name} plot saved to {metric_path}")
                    
            # Plot train vs evaluation metrics (comparison plots)
            if 'fit' in self.results['metrics']['distributed'] and 'evaluate' in self.results['metrics']['distributed']:
                train_metrics = self.results['metrics']['distributed']['fit']
                eval_metrics = self.results['metrics']['distributed']['evaluate']
                
                # Find common metrics between training and evaluation
                common_metrics = set(train_metrics.keys()).intersection(set(eval_metrics.keys()))
                
                for metric_name in common_metrics:
                    plt.figure(figsize=(12, 7))
                    
                    # Extract rounds and values from training data
                    train_rounds = [item[0] for item in train_metrics[metric_name]]
                    train_values = [item[1] for item in train_metrics[metric_name]]
                    
                    # Extract rounds and values from evaluation data
                    eval_rounds = [item[0] for item in eval_metrics[metric_name]]
                    eval_values = [item[1] for item in eval_metrics[metric_name]]
                    
                    # Plot both lines
                    plt.plot(train_rounds, train_values, 'o-', linewidth=2, markersize=8, 
                             color='blue', label=f'Training {metric_name}')
                    plt.plot(eval_rounds, eval_values, 'o-', linewidth=2, markersize=8, 
                             color='green', label=f'Evaluation {metric_name}')
                    
                    plt.title(f'Training vs Evaluation {metric_name}', fontsize=16)
                    plt.xlabel('Round', fontsize=14)
                    plt.ylabel(metric_name, fontsize=14)
                    plt.grid(True, alpha=0.3)
                    plt.xticks(train_rounds)
                    plt.legend(fontsize=12)
                    plt.tight_layout()
                    
                    # Save the figure
                    metric_path = os.path.join(fig_directory, f'compare_{metric_name}.png')
                    plt.savefig(metric_path)
                    plt.close()
                    print(f"Comparison {metric_name} plot saved to {metric_path}")
    
    def print_results_table(self) -> None:
        """Display the results in a table format."""
        if not self.results:
            print("No results loaded. Call load_simulation_results first.")
            return
        
        # Create table
        table = PrettyTable()
        table.field_names = ["Round", "Loss"] + self.metric_names
        
        # Get loss values
        loss_values = {}
        if 'loss' in self.results and 'distributed' in self.results['loss']:
            loss_values = {i+1: val for i, val in enumerate(self.results['loss']['distributed'])}
        
        # Get metric values (training and evaluation)
        metric_values = {metric: {} for metric in self.metric_names}
        
        if 'metrics' in self.results and 'distributed' in self.results['metrics']:
            # Training metrics
            if 'fit' in self.results['metrics']['distributed']:
                for metric, values in self.results['metrics']['distributed']['fit'].items():
                    for round_num, value in values:
                        if metric in metric_values:
                            metric_values[metric][round_num] = value
            
            # Evaluation metrics
            if 'evaluate' in self.results['metrics']['distributed']:
                for metric, values in self.results['metrics']['distributed']['evaluate'].items():
                    for round_num, value in values:
                        if metric in metric_values:
                            metric_values[metric][round_num] = value
        
        # Fill table rows
        for round_num in range(1, self.rounds + 1):
            row = [round_num]
            
            # Add loss
            loss = loss_values.get(round_num, "N/A")
            row.append(f"{loss:.6f}" if isinstance(loss, (int, float)) else loss)
            
            # Add metrics
            for metric in self.metric_names:
                value = metric_values.get(metric, {}).get(round_num, "N/A")
                row.append(f"{value:.6f}" if isinstance(value, (int, float)) else value)
            
            table.add_row(row)
        
        print("\nFederated Learning Results Summary:")
        print(table)