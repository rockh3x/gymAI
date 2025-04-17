import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim


class PolicyNetwork(nn.Module):
    def __init__(self, obs_space, action_space):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_space, 128),  # Input layer
            nn.ReLU(),  # Activation layer
            nn.Linear(128, action_space),  # Output layer
            nn.Softmax(dim=-1)  # Output probabilities
        )

    def forward(self, x):
        return self.fc(x)


def train_local_env(env_name, server):
    """
    Train an RL agent locally in a Gymnasium environment and send updates to the server.
    """
    print(f"[train_local_env] Initializing environment: {env_name}")
    env = gym.make(env_name)
    obs_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    policy = PolicyNetwork(obs_space, action_space)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    total_rewards = []  # List to store total rewards per episode

    try:
        print("[train_local_env] Starting training...")
        for episode in range(10):  # Train for 10 episodes
            obs = env.reset(seed=42)[0]  # Reset environment
            rewards = []  # Track rewards for the episode
            log_probs = []  # Track log probabilities for actions
            done = False

            # Perform steps in the environment
            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # Convert observation to Tensor
                action_probs = policy(obs_tensor)  # Get action probabilities
                action = torch.multinomial(action_probs, num_samples=1)  # Sample action
                log_prob = torch.log(action_probs.squeeze(0)[action])

                obs, reward, done, truncated, _ = env.step(action.item())  # Step in environment
                rewards.append(reward)
                log_probs.append(log_prob)

            total_rewards.append(sum(rewards))  # Store total episode reward
            print(f"[train_local_env] Episode {episode + 1} completed with total reward: {sum(rewards)}")

        # Send updates to the server
        print("[train_local_env] Training complete. Sending updates to the server...")
        server.receive_updates([policy.state_dict()])

        # Receive the global model from the server
        global_weights = server.send_global_model()
        policy.load_state_dict(global_weights)
        print("[train_local_env] Global model weights updated.")
    except Exception as e:
        print(f"[train_local_env] Error during training: {e}")
        total_rewards = []

    return total_rewards  # Return rewards for this client
