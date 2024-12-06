import tensorflow as tf
import numpy as np
import gymnasium
from gymnasium import spaces
import matplotlib.pyplot as plt
import random

# Hyperparameters
gamma = 0.99  # Discount factor
lr_actor = 0.001  # Actor learning rate
lr_critic = 0.001  # Critic learning rate
clip_ratio = 0.2  # PPO clip ratio
epochs = 10  # Number of optimization epochs
batch_size = 64  # Batch size for optimization

# Job Scheduling Environment
class ReentrantNetworkEnv(gymnasium.Env):
    def __init__(self, distance_matrix, buffer_capacity=5, num_buffers=10, failure_prob=0.1, max_time_steps=100):
        super(ReentrantNetworkEnv, self).__init__()

        self.distance_matrix = distance_matrix
        self.num_buffers = num_buffers  # number of buffers
        self.buffer_capacity = buffer_capacity  # max packages per buffer
        self.failure_prob = failure_prob  # probability of machine failure
        self.max_time_steps = max_time_steps

        # State size: [# packages in buffer 1, distance of robot to buffer 1,..., # packages in buffer J, distance of robot to buffer J]
        self.state_size = 2 * num_buffers  # no. of packages in each buffer and distance of robot to each buffer
        self.action_size = num_buffers + 1  # actions: 1 to num_buffers - transport package at that buffer, 0 - idle

        # Define the observation space and action space
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.state_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.action_size)

        # Initialize environment state
        self.reset()

    def reset(self):
        """Resets the environment state at the beginning of an episode."""
        self.buffer = [[] for _ in range(self.num_buffers)]  # Initialize empty buffers
        self.robot_distances = self.distance_matrix[0]  # Robot starts at buffer 1
        self.failures = np.zeros(self.num_buffers)  # No failures initially
        self.time_step = 0  # Track time steps
        self.done = False  # Not done initially
        return np.array(self.get_state(), dtype=np.float32)

    def get_state(self):
        """Returns the current state of the environment."""
        package_counts = [len(self.buffer[i]) for i in range(self.num_buffers)]
        robot_distances = self.robot_distances
        state = np.concatenate([package_counts, robot_distances])
        return np.array(state, dtype=np.float32)

    def step(self, action):
        """Take a step in the environment based on the robot's action."""
        self.time_step += 1
        reward = 0

        # Handle action: move or idle
        if action == 0:  # robot in waiting position
            current_buffer = np.where(distance_matrix == self.robot_distances) # determine current buffer
            if current_buffer == 0: # Robot was last at buffer 1 and has to move to the right
                self.robot_distances = [distance_matrix[0][i] + 0.5 * (distance_matrix[1][i] - distance_matrix[0][i])
                                        for i in len(distance_matrix[0])] # calculate Average
            elif current_buffer == 9: # Robot was last at buffer 10 and has to move to the left
                self.robot_distances = [distance_matrix[9][i] - 0.5 * (distance_matrix[9][i] - distance_matrix[8][i])
                                        for i in len(distance_matrix[0])] # calculate Average
            else:
                left_or_right = random.randint(1,2) # determines whether robot goes left or right
                if left_or_right == 1: # if left
                    self.robot_distances = [distance_matrix[current_buffer][i] - 0.5 * (
                            distance_matrix[current_buffer][i] - distance_matrix[current_buffer - 1][i]) for i in
                                            len(distance_matrix[0])]  # calculate Average
                else: # if right
                    self.robot_distances = [distance_matrix[current_buffer][i] + 0.5 * (
                                distance_matrix[current_buffer + 1][i] - distance_matrix[current_buffer][i]) for i in
                                            len(distance_matrix[0])]  # calculate Average

        else: # Robot moving to buffer
            #print(f"Buffer {action - 1}: {self.buffer[action - 1]}, Capacity: {self.buffer_capacity}")
            self.robot_distances = self.distance_matrix[action - 1]  # Robot arrives at new buffer
            if len(self.buffer[action - 1]) > 0 and len(self.buffer[action - 1]) < self.buffer_capacity:
                self.buffer[action - 1].pop(0)  # Package leaves current buffer to be processed
                # simulate machine failures, which returns service time
                service_time = self._simulate_failures()
                if action < self.num_buffers:
                    self.buffer[action].append(self.time_step + service_time) # Package is appended at next buffer

        # Compute costs as a negative reward
        total_packages = sum([len(b) for b in self.buffer])
        reward -= total_packages

        # Simulate package arrivals
        self._package_arrivals()

        # Check if episode is done
        if self.time_step >= self.max_time_steps:
            self.done = True

        # Get the next state after the action
        next_state = np.array(self.get_state(), dtype=np.float32)
        return next_state, reward, self.done, {}

    def _package_arrivals(self):
        """Simulates the arrival of new packages into buffer 1."""
        new_packages = np.random.poisson(1)  # Avg 1 new package per time step
        for _ in range(new_packages):
            if len(self.buffer[0]) < self.buffer_capacity: # Add package to first buffer if there is capacity
                self.buffer[0].append(self.time_step)

    def _simulate_failures(self):
        """Simulates random failures in the machines."""
        for i in range(self.num_buffers):
            if np.random.rand() < self.failure_prob:
                self.failures[i] = 1
                if len(self.buffer[i]) > 0:
                    self.buffer[i].pop(0)  # Pop first package in buffer
                    service_time = 20 + random.randint(0, 10)/100  # return service time with penalty
            else:
                self.failures[i] = 0
                service_time = random.randint(0,10)/100 # return normal service time
        return service_time

# PPO Actor-Critic Model
class ActorCritic(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.policy_logits = tf.keras.layers.Dense(action_size)
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.value = tf.keras.layers.Dense(1)

    def call(self, state):
        x = self.dense1(state)
        logits = self.policy_logits(x)
        value = self.value(x)
        return logits, value


# PPO Loss Function and Training Step
def ppo_loss(old_logits, old_values, advantages, states, actions, returns):
    def compute_loss(logits, values, actions, returns):
        actions_onehot = tf.one_hot(actions, action_size, dtype=tf.float32)
        policy = tf.nn.softmax(logits)
        action_probs = tf.reduce_sum(actions_onehot * policy, axis=1)
        old_policy = tf.nn.softmax(old_logits)
        old_action_probs = tf.reduce_sum(actions_onehot * old_policy, axis=1)

        # Policy loss
        ratio = tf.exp(tf.math.log(action_probs + 1e-10) - tf.math.log(old_action_probs + 1e-10))
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        policy_loss = tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

        # Value loss
        value_loss = tf.reduce_mean(tf.square(values - returns))

        # Entropy bonus (optional)
        entropy_bonus = tf.reduce_mean(policy * tf.math.log(policy + 1e-10))

        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
        return total_loss

    def get_advantages(returns, values):
        advantages = returns - values
        return (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

    def train_step(states, actions, returns, old_logits, old_values):
        with tf.GradientTape() as tape:
            logits, values = model(states)
            loss = compute_loss(logits, values, actions, returns)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    advantages = get_advantages(returns, old_values)
    for _ in range(epochs):
        loss = train_step(states, actions, returns, old_logits, old_values)
    return loss


# Initialize environment and model
distance_matrix = [
        [0, 2, 4, 7, 8, 10, 13, 15, 16, 19],  # Distances from buffer 1 to buffers 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        [2, 0, 3, 5, 7, 9, 12, 13, 14, 15],  # Distances from buffer 2 to buffers 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        [4, 3, 0, 2, 3, 5, 8, 10, 11, 12],   # Distances from buffer 3 to buffers 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        [7, 5, 2, 0, 1, 4, 5, 8, 9, 11], # Distances from buffer 4 to buffers 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        [8, 7, 3, 1, 0, 3, 4, 7, 8, 9], # Distances from buffer 5 to buffers 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        [10, 9, 5, 4, 3, 0, 1, 3, 5, 6], # Distances from buffer 6 to buffers 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        [13, 12, 8, 5, 4, 1, 0, 2, 4, 7], # Distances from buffer 7 to buffers 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        [15, 13, 10, 8, 7, 3, 2, 0, 3, 5], # Distances from buffer 8 to buffers 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        [16, 14, 11, 9, 8, 5, 4, 3, 0, 2], # Distances from buffer 9 to buffers 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        [19, 15, 12, 11, 9, 6, 7, 5, 2, 0], # Distances from buffer 10 to buffers 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    ]
distance_matrix = np.array(distance_matrix)

env = ReentrantNetworkEnv(distance_matrix)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
model = ActorCritic(state_size, action_size)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Main Training Loop with Cost Tracking
max_episodes = 200
max_steps_per_episode = 100
costs_per_episode = []


for episode in range(max_episodes):
    states, actions, rewards, values = [], [], [], []
    state = env.reset()
    for step in range(max_steps_per_episode):
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        logits, value = model(state)

        # Sample action from the policy distribution
        action = tf.random.categorical(logits, 1)[0, 0].numpy()
        next_state, reward, done, _ = env.step(action)

        # Collect data
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        values.append(value)

        state = next_state

        if done or step == max_steps_per_episode - 1:
            returns_batch = []
            discounted_sum = 0
            for r in rewards[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns_batch.append(discounted_sum)
            returns_batch.reverse()

            # Prepare data for training
            states = tf.concat(states, axis=0)
            actions = np.array(actions, dtype=np.int32)
            values = tf.concat(values, axis=0)
            returns_batch = tf.convert_to_tensor(returns_batch)
            old_logits, _ = model(states)

            # Train the model
            loss = ppo_loss(old_logits, values, returns_batch - np.array(values), states, actions, returns_batch)

            average_cost = -np.mean(rewards)
            costs_per_episode.append(average_cost)
            break

# Plotting the costs over episodes
plt.plot(costs_per_episode)
plt.xlabel('Episodes')
plt.ylabel('Average Holding Cost')
plt.title('Average Holding Cost over Episodes')
plt.show()