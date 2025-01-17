import matplotlib.pyplot as plt
import numpy as np

# Function to parse the input text file
def parse_data(filename):
    data = {}
    with open(filename, 'r') as file:
        lines = file.readlines()

    generation = -1
    for line in lines:
        line = line.strip()
        if line.endswith("Generations:"):
            generation = int(line.split()[0])
            data[generation] = []
        elif line.startswith("Cantidate"):
            candidate_data = line.split(":")[1]
            data[generation].append(candidate_data)
    return data

# Function to plot the data
def plot_data(data):
    generations = list(data.keys())
    num_generations = len(generations)
    num_candidates = len(next(iter(data.values())))

    plt.figure(figsize=(10, 6))

    # Plot each candidate's binary values across generations
    for candidate_idx in range(num_candidates):
        # Extract binary strings
        binary_values = [data[gen][candidate_idx] for gen in generations]

        # Plot binary values as strings (we'll plot their index on the y-axis)
        plt.plot(generations, [int(b, 2) for b in binary_values], marker='o', label=f"Candidate {candidate_idx}")

        # Annotate binary values at each point
        for gen_idx, binary_value in enumerate(binary_values):
            plt.text(generations[gen_idx], int(binary_value, 2), f"{binary_value}",
                     ha='center', va='center', fontsize=8,
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # Set axis labels, title, and legend
    plt.title("Binary Values Evolution Across Generations by BinVal", fontsize=16)
    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Binary Value", fontsize=14)
    plt.xticks(generations)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # Save the plot instead of showing it interactively
    plt.savefig("candidates_progression_plot.png", bbox_inches="tight")
    print("Plot saved as 'candidates_progression_plot.png'.")

# Main script
if __name__ == "__main__":
    filename = "Results.txt"  # Replace with your file name
    data = parse_data(filename)
    plot_data(data)
