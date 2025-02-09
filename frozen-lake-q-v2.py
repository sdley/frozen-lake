import gymnasium as gym  # Import the Gymnasium library for environment creation
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import pickle  # Import Pickle for saving and loading Q-tables

def run(episodes, is_training=True, render=True):
    """
    Runs the Q-learning algorithm on the FrozenLake environment.

    Args:
        episodes (int): The number of episodes to run.
        is_training (bool): Whether to train the Q-table or load a pre-trained one.
        render (bool): Whether to render the environment during execution.
    """

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='human' if render else None)
    # Create the FrozenLake environment.  'map_name' specifies the size, 'is_slippery' disables stochastic transitions,
    # and 'render_mode' enables visualization if render is True.

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array
    # If training, initialize the Q-table with zeros.  The Q-table has dimensions
    # (number of states x number of actions).  For 8x8, this is 64x4.
    else:
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()
    # If not training, load the Q-table from a file.

    learning_rate_a = 0.9 # alpha or learning rate
    # Set the learning rate (alpha).  This controls how much the Q-table is updated
    # with each new experience.
    discount_factor_g = 0.9 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. 
    # Near 1: more on future state.
    # Set the discount factor (gamma).  This controls the importance of future rewards.
    epsilon = 1         # 1 = 100% random actions
    # Set the initial exploration rate (epsilon).  This controls the probability of
    # taking a random action to explore the environment.
    epsilon_decay_rate = 0.0001        # epsilon decay rate. 1/0.0001 = 10,000
    # Set the epsilon decay rate.  This reduces epsilon over time, decreasing exploration.
    rng = np.random.default_rng()   # random number generator
    # Create a random number generator.

    rewards_per_episode = np.zeros(episodes)
    # Initialize an array to store the rewards for each episode.

    for i in range(episodes):
        state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        # Reset the environment and get the initial state.  States are numbered from 0 to 63.
        terminated = False      # True when fall in hole or reached goal
        # Initialize the 'terminated' flag, which indicates whether the episode has ended
        # because the agent reached the goal or fell into a hole.
        truncated = False       # True when actions > 200
        # Initialize the 'truncated' flag, which indicates whether the episode has ended
        # because the maximum number of steps (200) has been reached.

        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
            # If training and a random number is less than epsilon, take a random action.
            # Actions are: 0=left, 1=down, 2=right, 3=up.
            else:
                action = np.argmax(q[state,:])
            # Otherwise, take the action with the highest Q-value for the current state.

            new_state,reward,terminated,truncated,_ = env.step(action)
            # Take the chosen action and observe the next state, reward, and whether
            # the episode has terminated or been truncated.

            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )
            # If training, update the Q-table using the Q-learning update rule.
            # Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a))

            state = new_state
            # Update the current state to the next state.

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        # Decrease epsilon by the decay rate, but don't let it go below 0.

        if(epsilon==0):
            learning_rate_a = 0.0001
        # After epsilon reaches 0, reduce the learning rate

        if reward == 1:
            rewards_per_episode[i] = 1
        # If the episode ended with a reward of 1 (reached the goal), record it.

    env.close()
    # Close the environment.

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake8x8.png')
    # Calculate the moving average of the rewards over the last 100 episodes and plot it.

    if is_training:
        f = open("frozen_lake8x8.pkl","wb")
        pickle.dump(q, f)
        f.close()
    # If training, save the trained Q-table to a file.

if __name__ == '__main__':
    # run(15000)

    run(100, is_training=True, render=True)
    # Run the training with 1000 episodes, training mode on, and rendering enabled
