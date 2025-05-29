import matplotlib.pyplot as plt
import numpy as np
import os


def objective_circuit_plot(graphs_str, result_data, color, name_str, PLOT_DIR):
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, data in enumerate(result_data):
        ax.plot(
            data["objective_func_cost"],
            marker='o',
            linestyle='-',
            color=color[i],
            label=graphs_str[i]
        )

    ax.set_title('Convergence of the expected energy ⟨H^⟩ across QAOA: ' + name_str)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Energy')
    ax.legend()

    os.makedirs(PLOT_DIR, exist_ok=True)
    fig.savefig(os.path.join(PLOT_DIR, f"objective_plot_{name_str}.png"), bbox_inches="tight")

    return fig


def average_plot(graphs_str, result_data, color, name_str):

    for i, data in enumerate(result_data):
        # Extract the list of sample distributions
        sample_distributions = data["sample_distribution"]

        # Estimate variance for each sample distribution
        estimated_variances = [np.average(list(dist.values())) for dist in sample_distributions]

        # Plotting the estimated variance over iterations
        plt.plot(range(len(estimated_variances)), estimated_variances, color=color[i], label=graphs_str[i])

    plt.xlabel("Iteration")
    plt.ylabel("Average")
    plt.title("Convergence of Average: " + name_str)
    #plt.grid(True)
    plt.show()

def average_energy_plot(graphs_str, result_data, color, name_str, hamiltonians, calculate_cost):

    estimated_averages = []
    j = 0
    for i, data in enumerate(result_data):
        # Extract the distribution of samples
        sample_distribution = data["sample_distribution"]

        # Sort samples by frequency (optional)
        for samples in sample_distribution:
            top_items = sorted(samples.items(), key=lambda item: item[1], reverse=True)

            all_costs = []

            for sample, freq in top_items:
                sample_int = int(sample,2)
                cost = calculate_cost(sample_int, hamiltonians[j])
                all_costs.append(cost)

            # Calculate average cost
            avg_cost = np.mean(all_costs[:10])
            estimated_averages.append(avg_cost)


        j += 1
        # Print optional details
        #print(f"[{graphs_str[i]}] Avg Cost (All): {avg_cost:.4f} from {len(all_costs)} samples")

        # Optional: Top 10 average
        if len(all_costs) >= 10:
            avg_top_10 = np.mean(all_costs[:10])
            #print(f"[{graphs_str[i]}] Avg Cost (Top 10): {avg_top_10:.4f}")

        # Plot the average cost per graph
        plt.plot(range(len(estimated_averages)), estimated_averages, color=color[i], label=graphs_str[i])

        estimated_averages = []

    plt.xlabel("Iteration")
    plt.ylabel("Average Cost")
    plt.title("Objective function MaxCut Top 10: " + name_str)
    plt.legend()
    plt.show()

def varianza_estimation_plot(graphs_str, result_data, color, name_str):

    for i, data in enumerate(result_data):
        # Extract the list of sample distributions
        sample_distributions = data["sample_distribution"]

        # Estimate variance for each sample distribution
        estimated_variances = [np.var(list(dist.values())) for dist in sample_distributions]

        # Plotting the estimated variance over iterations
        plt.plot(range(len(estimated_variances)), estimated_variances, color=color[i], label=graphs_str[i])

    plt.xlabel("Iteration")
    plt.ylabel("Estimated Variance")
    plt.title("Convergence of Variance Estimation: " + name_str)
    #plt.grid(True)
    plt.show()