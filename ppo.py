import tensorflow as tf
import numpy as np
import gymnasium
from gymnasium import spaces
import matplotlib.pyplot as plt
import random
import time

# Hyperparameters
lr_actor = 0.00001
lr_critic = 0.0001
clip_ratio = 0.2
epochs = 10
batch_size = 32
gamma = 1.0

# Job Scheduling Environment
class ReentrantNetworkEnv(gymnasium.Env):
    def __init__(self, distance_matrix, buffer_capacity_1 = 100, buffer_capacity=10, num_buffers=5, failure_prob=0.00, max_time_steps=1000):
        ''' Initialize the Reentrant Network Environment '''
        super(ReentrantNetworkEnv, self).__init__()

        self.distance_matrix = distance_matrix
        self.num_buffers = num_buffers + 1 # last buffer represents the machine of last buffer to track ready time
        self.buffer_capacity_1 = buffer_capacity_1
        self.buffer_capacity = buffer_capacity
        self.failure_prob = failure_prob
        self.max_time_steps = max_time_steps

        self.state_size = 2 * num_buffers
        self.action_size = num_buffers + 1 # action for each buffer and one for wait action

        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.state_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.action_size)

        self.trigger_buffer = []
        self.packages_left = 0
        self.reset()

    def reset(self):
        ''' Reset the environment to the initial state '''
        self.buffer = [[] for _ in range(self.num_buffers)]
        self.robot_distances = self.distance_matrix[0]
        self.failures = np.zeros(self.num_buffers)
        self.time_step = 0
        self.done = False
        self.trigger_buffer.clear()
        self.packages_left = 0
        return self.get_state()

    def step(self, action):
        ''' Simulate the environment for one step depending on the action chosen'''
        self.time_step += 1
        reward = 0

        # at each time step, check if package at last machine is ready to leave the system
        if 0 < len(self.buffer[-1]) and self.buffer[-1][0][1] <= self.time_step:
            self.buffer[-1].pop(0) # remove package from last machine
            self.packages_left += 1
            print("Package leaves system")

        if 0 < action < self.num_buffers - 1: # if any action but last buffer is chosen
            if 0 < len(self.buffer[action - 1]) and len(self.buffer[action]) < self.buffer_capacity and all(ready_time <= self.time_step for _, ready_time in self.buffer[action - 1]): # if current buffer is not empty and new buffer not full and machine not busy
                if self.buffer[action - 1][0][1] <= self.time_step: # if package is ready
                    self.buffer[action - 1].pop(0) # remove package from current buffer (FIFO)
                    service_time = self._service_time(buffer_index = action - 1) # call service time function
                    ready_time = self.time_step + service_time # calculate ready time of package
                    self.buffer[action].append((self.time_step, ready_time))

        elif action == self.num_buffers - 1:  # if action is at last buffer
            if 0 < len(self.buffer[action - 1]) and len(self.buffer[action]) < 1: # allow only one package in last machine
                self.buffer[action - 1].pop(0)  # remove package from last buffer (FIFO)
                service_time = self._service_time(buffer_index=action - 1)  # call service time function
                ready_time = self.time_step + service_time  # calculate ready time of package
                self.buffer[action].append((self.time_step, ready_time))

        # if action is 0, do nothing

        self.robot_distances = self.distance_matrix[action - 1]  # set new location of robot at current buffer

        total_packages = sum(len(b) for b in self.buffer)
        reward -= total_packages

        self._package_arrivals()
        self._trigger_decision_epoch()

        if self.time_step >= self.max_time_steps:
            self.done = True

        next_state = self.get_state()
        return next_state, reward, self.done, {}

    def _package_arrivals(self):
        ''' Simulate actual and virtual package arrivals to the system '''
        new_packages = np.random.poisson(0.1) # arrival rate of packages
        for _ in range(new_packages):
            if len(self.buffer[0]) < self.buffer_capacity_1: # if first buffer is not full
                self.buffer[0].append((self.time_step, self.time_step)) # add package to first buffer (ready immediately)

        virtual_arrival = np.random.poisson(0.3) # simulate virtual arrivals to wait buffer
        if virtual_arrival > 0:
            self.trigger_buffer.append(self.time_step)

    def _trigger_decision_epoch(self):
        ''' Trigger when time_step reaches time of arrival of virtual packages '''
        if self.trigger_buffer and self.trigger_buffer[0] <= self.time_step:
            self.trigger_buffer.pop(0)

    def _service_time(self, buffer_index):
        ''' Calculate service time for a package at a buffer'''
        service_time = float(0.001 + self.robot_distances[buffer_index]) # service time plus transportation time
        if np.random.rand() < self.failure_prob: # simulate failure with probability failure_prob
            self.failures[buffer_index] = 1 # track failures
            service_time = float(service_time + 0.1) # add penalty for failure
            if self.buffer[buffer_index] and self.buffer[buffer_index][0][1] <= self.time_step: # if package is ready
                self.buffer[buffer_index].pop(0) # remove package from buffer
        else:
            self.failures[buffer_index] = 0
        return service_time

    def get_state(self):
        package_counts = [len(self.buffer[i]) for i in range(self.num_buffers)]
        robot_distances = self.robot_distances
        state = np.concatenate([package_counts, robot_distances])
        return state.astype(np.float32)

class ActorCritic(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.policy_logits = tf.keras.layers.Dense(action_size)
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.value = tf.keras.layers.Dense(1)

    def call(self, state):
        x = self.dense1(state)
        logits = self.policy_logits(x)
        value = self.value(x)
        return logits, value

#PPO Algorithm
def ppo_loss(model, optimizer, old_logits, old_values, advantages, states, actions, returns):
    action_size = model.policy_logits.units

    def compute_loss(logits, values, actions, returns):
        actions_onehot = tf.one_hot(actions, action_size, dtype=tf.float32)
        policy = tf.nn.softmax(logits)
        action_probs = tf.reduce_sum(actions_onehot * policy, axis=1)
        old_policy = tf.nn.softmax(old_logits)
        old_action_probs = tf.reduce_sum(actions_onehot * old_policy, axis=1)

        ratio = tf.exp(tf.math.log(action_probs + 1e-10) - tf.math.log(old_action_probs + 1e-10))
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        policy_loss = tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

        value_loss = tf.reduce_mean(tf.square(values - returns))

        entropy_bonus = tf.reduce_mean(-policy * tf.math.log(policy + 1e-10))

        total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_bonus # 0.5 and 0.01 are hyperparameters
        return total_loss

    def get_advantages(returns, values):
        advantages = (returns - old_values)
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

def discount_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    cumulative = 0.0
    for i in reversed(range(len(rewards))):
        cumulative = cumulative * gamma + rewards[i]
        discounted_rewards[i] = cumulative
    return discounted_rewards

# Main training loop
distance_matrix = np.array([
    [0, 1/19, 1/16, 1/15, 1/13, 1/10, 1/8, 1/7, 1/4, 1/2],
    [1/19, 0, 1/17, 1/16, 1/14, 1/12, 1/9, 1/8, 1/5, 1/3],
    [1/16, 1/17, 0, 1/18, 1/16, 1/13, 1/11, 1/10, 1/8, 1/4],
    [1/15, 1/16, 1/18, 0, 1/18, 1/14, 1/12, 1/11, 1/9, 1/5],
    [1/13, 1/14, 1/16, 1/18, 0, 1/16, 1/13, 1/12, 1/10, 1/6],
    [1/10, 1/12, 1/12, 1/14, 1/16, 0, 1/15, 1/14, 1/12, 1/8],
    [1/8, 1/9, 1/11, 1/12, 1/13, 1/15, 0, 1/18, 1/17, 1/13],
    [1/7, 1/8, 1/10, 1/11, 1/12, 1/14, 1/18, 0, 1/18, 1/14],
    [1/4, 1/5, 1/8, 1/9, 1/10,1/12, 1/17, 1/18, 0, 1/15],
    [1/2, 1/3, 1/4, 1/5, 1/6, 1/8, 1/13, 1/14, 1/15, 0],
], dtype=float)
distance_matrix = distance_matrix / 100

start_time = time.time()

env = ReentrantNetworkEnv(distance_matrix)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
model = ActorCritic(state_size, action_size)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

max_episodes = 200
max_steps_per_episode = 1000
average_rewards_per_episode = []
packages_left_per_episode = []

for episode in range(max_episodes):
    states, actions, rewards, values = [], [], [], []
    discounted_reward = 0
    state = env.reset()
    done = False

    for step in range(max_steps_per_episode):
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        logits, value = model(state)
        action = tf.random.categorical(logits, 1)[0, 0].numpy()

        next_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        values.append(value)

        state = next_state

        if done:
            break

    rewards = discount_rewards(rewards, gamma)
    returns_batch = tf.convert_to_tensor(rewards, dtype=tf.float32)

    average_rewards_per_episode.append(-1 * (rewards[0]/len(rewards)))

    states = tf.concat(states, axis=0)
    actions = np.array(actions, dtype=np.int32)
    values = tf.concat(values, axis=0)
    old_logits, _ = model(states)

    advantages = returns_batch - tf.squeeze(values)

    loss = ppo_loss(
        model=model,
        optimizer=optimizer,
        old_logits=old_logits,
        old_values=values,
        advantages=advantages,
        states=states,
        actions=actions,
        returns=returns_batch
    )

    packages_left_per_episode.append(env.packages_left)

end_time = time.time()
training_duration = end_time - start_time

fig, axes = plt.subplots(1,2)#
fig.suptitle('Training Progress')
axes[0].plot(average_rewards_per_episode)
axes[0].set_xlabel('Episodes')
axes[0].set_ylabel('Average Amount of Packages')
#axes[0].set_title('Average Amount of Packages over Episodes')
axes[1].plot(packages_left_per_episode)
axes[1].set_xlabel('Episodes')
axes[1].set_ylabel('Number of Packages Left')
#axes[1].set_title('Number of Packages Leaving the System per Episode')
plt.show()



print(f"Training completed in {training_duration:.2f} seconds")