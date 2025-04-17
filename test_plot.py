import matplotlib.pyplot as plt

rewards = [10, 20, 15, 25, 30]  # Example data
plt.plot(rewards, label="Example Client")
plt.title("Test Plot")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend()
plt.grid()
plt.show()
