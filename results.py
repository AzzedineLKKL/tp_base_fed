import os
import sys
from results_vis import ResultsVisualizer

def main():
    """Analyze federated learning results."""
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python analyze_results.py <results_file.json>")
        return
    
    results_file = sys.argv[1]
    if not os.path.exists(results_file):
        print(f"Error: Results file '{results_file}' not found")
        return
    
    # Create visualizer
    visualizer = ResultsVisualizer()
    
    # Load results
    print(f"Loading results from {results_file}...")
    visualizer.load_simulation_results(results_file)
    
    # Print results table
    visualizer.print_results_table()
    
    # Create plots
    fig_dir = "result_figures"
    print(f"Creating plots in directory: {fig_dir}")
    visualizer.plot_results(fig_dir)
    
    print(f"\nAnalysis complete. Visualizations saved to {fig_dir}/")

if __name__ == "__main__":
    main()