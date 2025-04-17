from federated_server import FederatedServer
from local_environment import train_local_env
import matplotlib.pyplot as plt


def main():
    print("Execution started!")
    print("This is the main.py script!")

    # Initialize the federated server
    print("Initializing Federated Server...")
    server = FederatedServer()
    print("Federated Server initialized!")

    # Federated learning configuration
    env_name = "CartPole-v1"
    num_clients = 5
    print(f"Starting Federated Learning setup for {num_clients} clients with environment '{env_name}'")

    # Collect training rewards
    client_rewards = []  # List to store rewards for all clients
    for i in range(num_clients):
        print(f"\nClient {i + 1}: Starting training...")
        try:
            # Train client and retrieve rewards
            rewards = train_local_env(env_name, server)
            
            # Check if rewards are valid before appending
            if rewards and isinstance(rewards, list) and all(isinstance(r, (int, float)) for r in rewards):
                client_rewards.append(rewards)
                print(f"Client {i + 1} rewards: {rewards}")
            else:
                print(f"Warning: Client {i + 1} returned invalid or empty rewards.")
                client_rewards.append([])  # Add an empty list for consistency
        except Exception as e:
            print(f"Error during training for Client {i + 1}: {e}")
            client_rewards.append([])  # Add an empty list for failed clients

        print(f"Client {i + 1}: Training completed!")

    print("\nFederated Learning Implementation Complete!")

    # Debug: Print summary of all client rewards
    print("\nFinal Client Rewards Summary:")
    for client_id, rewards in enumerate(client_rewards, start=1):
        print(f"Client {client_id} Rewards: {rewards}")

    # Plot the rewards
    plot_rewards(client_rewards)


def plot_rewards(client_rewards):
    """Plot the rewards for all clients."""
    plt.figure(figsize=(10, 6))

    # Debugging: Print rewards being processed
    print("\nDEBUG: Preparing to plot rewards for each client.")

    has_valid_data = False  # Track if any valid rewards exist for plotting

    for client_id, rewards in enumerate(client_rewards, start=1):
        # Validate rewards before plotting
        if rewards and isinstance(rewards, list) and all(isinstance(r, (int, float)) for r in rewards):
            print(f"DEBUG: Plotting rewards for Client {client_id}: {rewards}")  # Debug line
            try:
                plt.plot(rewards, label=f"Client {client_id}")  # Plot only valid data
                has_valid_data = True
            except Exception as e:
                print(f"ERROR: Failed to plot rewards for Client {client_id}. Error: {e}")
        else:
            print(f"WARNING: Invalid rewards for Client {client_id}. Skipping. Rewards: {rewards}")

    if has_valid_data:
        plt.title("Total Reward per Episode for Each Client")
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid()
        plt.show()  # Render the plot
    else:
        print("No valid rewards to plot. Exiting without rendering a plot.")


if __name__ == "__main__":
    main()
