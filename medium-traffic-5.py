import tensorflow as tf
import numpy as np
import gymnasium
from gymnasium import spaces
import matplotlib.pyplot as plt
import random
import time

# Hyperparameters
lr_actor = 0.0001
lr_critic = 0.001
clip_ratio = 0.2
epochs = 10
batch_size = 32
gamma = 1.0

# Job Scheduling Environment
class ReentrantNetworkEnv(gymnasium.Env):
    def __init__(self, distance_matrix, buffer_capacity_1 = 500, buffer_capacity=10, num_buffers=5, failure_prob=0.01, max_time_steps=1000):
        ''' Initialize the Reentrant Network Environment '''
        super(ReentrantNetworkEnv, self).__init__()

        self.distance_matrix = distance_matrix
        self.num_buffers = num_buffers
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
        self.machines_busy = [[] for _ in range(self.num_buffers)]
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
        cost = 0

        # at each time step, check if package at any machine but the last is ready to move to next buffer
        for i in range(len(self.machines_busy)-1):
            if 0 < len(self.machines_busy[i]) and self.machines_busy[i][0][0] <= self.time_step:
                if self.machines_busy[i][0][1]: # if the machine did not fail
                    self.machines_busy[i].pop(0)
                    self.buffer[i + 1].append(self.time_step)  # append package to next buffer
                else: # if the machine failed
                    self.machines_busy[i].pop(0) # pop package out of machine but do not append to next buffer


        # at each time step, check if package at last machine is ready to leave the system
        if 0 < len(self.machines_busy[-1]) and self.machines_busy[-1][0][0] <= self.time_step:
            if self.machines_busy[-1][0][1]: # if the machine did not fail
                self.machines_busy[-1].pop(0) # remove package from last machine
                self.packages_left += 1
            else: # if the machine failed
                self.machines_busy[-1].pop(0) # remove package from last machine, but do not count as leaving system

        if 0 < action < self.num_buffers: # if any action but last buffer is chosen
            if 0 < len(self.buffer[action - 1]) and len(self.buffer[action]) < self.buffer_capacity and len(self.machines_busy[action-1]) < 1: # if current buffer is not empty and new buffer not full and machine not busy
                self.buffer[action - 1].pop(0) # remove package from current buffer (FIFO)
                service_return = self._service_time(buffer_index = action - 1) # call service time function
                service_time = service_return[0]
                ready_time = self.time_step + service_time # calculate ready time of package
                self.machines_busy[action - 1].append((ready_time, service_return[1])) # add package to machine

        elif action == self.num_buffers:  # if action is at last buffer
            if 0 < len(self.buffer[action - 1]) and len(self.machines_busy[action-1]) < 1: # if current buffer is not empty and last machine not busy
                self.buffer[action - 1].pop(0)  # remove package from last buffer (FIFO)
                service_return = self._service_time(buffer_index=action - 1)  # call service time function
                service_time = service_return[0]
                ready_time = self.time_step + service_time  # calculate ready time of package
                self.machines_busy[action - 1].append((ready_time, service_return[1]))  # add package to machine

        if action > 0:
            self.robot_distances = self.distance_matrix[action - 1]  # set new location of robot at current buffer

        # if action is 0, do nothing

        total_packages = sum(len(b) for b in self.buffer)
        cost += total_packages

        self._package_arrivals()
        self._trigger_decision_epoch()

        if self.time_step >= self.max_time_steps:
            self.done = True

        next_state = self.get_state()
        return next_state, cost, self.done, {}

    def _package_arrivals(self):
        ''' Simulate actual and virtual package arrivals to the system '''
        new_packages = np.random.poisson(0.4) # arrival rate of packages
        for _ in range(new_packages):
            if len(self.buffer[0]) < self.buffer_capacity_1: # if first buffer is not full
                self.buffer[0].append(self.time_step) # add package to first buffer (ready immediately)

        virtual_arrival = np.random.poisson(0.2) # simulate virtual arrivals to wait buffer
        if virtual_arrival > 0:
            self.trigger_buffer.append(self.time_step)

    def _trigger_decision_epoch(self):
        ''' Trigger when time_step reaches time of arrival of virtual packages '''
        if self.trigger_buffer and self.trigger_buffer[0] <= self.time_step:
            self.trigger_buffer.pop(0)

    def _service_time(self, buffer_index):
        """Calculate service time for a package at a buffer."""
        # Generate a stochastic service time using Weibull distribution
        shape_param = 1.3  # shape parameter for the Weibull distribution
        scale_param = 0.1  # scale parameter
        stochastic_time = np.random.weibull(shape_param) * scale_param

        service_time = float(
            stochastic_time + self.robot_distances[
                buffer_index])  # Total Service time is processing time plus transportation time

        if np.random.rand() < self.failure_prob:  # Simulate failure with probability failure_prob
            self.failures[buffer_index] = 1  # Track failures
            service_time = 2 * service_time  # double service time as penalty
            success = False
        else:
            self.failures[buffer_index] = 0
            success = True

        return service_time, success

    def get_state(self):
        package_counts = [len(self.buffer[i]) for i in range(self.num_buffers)]
        robot_distances = self.robot_distances
        state = np.concatenate([package_counts, robot_distances])
        return state.astype(np.float32)


class ActorCritic(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.actor_layer = tf.keras.layers.Dense(32, activation='relu')  # Actor-specific layer
        self.critic_layer = tf.keras.layers.Dense(32, activation='relu')  # Critic-specific layer
        self.actor_output = tf.keras.layers.Dense(action_size, activation=None)  # Policy logits
        self.critic_output = tf.keras.layers.Dense(1)  # Value output


    def call(self, state):
        actor_x = self.actor_layer(state)
        critic_x = self.critic_layer(state)
        logits = self.actor_output(actor_x)  # Actor output (policy logits)
        value = self.critic_output(critic_x)  # Critic output (value estimate)
        return logits, value


# PPO Loss Function

def ppo_loss(model, optimizer_actor, optimizer_critic, old_logits, old_values, advantages, states, actions, returns):
    action_size = model.actor_output.units

    def compute_loss(logits, values, actions, returns):
        actions_onehot = tf.one_hot(actions, action_size, dtype=tf.float32)
        policy = tf.nn.softmax(logits)
        action_probs = tf.reduce_sum(actions_onehot * policy, axis=1)
        old_policy = tf.nn.softmax(old_logits)
        old_action_probs = tf.reduce_sum(actions_onehot * old_policy, axis=1)

        # Policy Loss
        ratio = tf.exp(tf.math.log(action_probs + 1e-10) - tf.math.log(old_action_probs + 1e-10))
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * - advantages, clipped_ratio * - advantages))

        # Value Loss
        value_loss = tf.reduce_mean(tf.square(values - returns))

        # Entropy Bonus
        entropy_bonus = tf.reduce_mean(-policy * tf.math.log(policy + 1e-10))

        total_loss = policy_loss + 0.5 * value_loss - 0.8 * entropy_bonus
        return total_loss, policy_loss, value_loss

    @tf.function
    def train_step(states, actions, returns, old_logits, old_values, advantages):
        with tf.GradientTape() as tape:
            logits, values = model(states, training=True)

            # Compute losses
            total_loss, policy_loss, value_loss = compute_loss(logits, values, actions, returns)

        # Compute and apply gradients
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer_actor.apply_gradients(zip(gradients, model.trainable_variables))
        optimizer_critic.apply_gradients(zip(gradients, model.trainable_variables))

        return policy_loss, value_loss

    for _ in range(epochs):
        policy_loss, value_loss = train_step(states, actions, returns, old_logits, old_values, advantages)

    return policy_loss, value_loss

def discount_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    cumulative = 0.0
    for i in reversed(range(len(rewards))):
        cumulative = cumulative * gamma + rewards[i]
        discounted_rewards[i] = cumulative
    return discounted_rewards

def normalize_returns(returns):
    mean = np.mean(returns)
    std = np.std(returns) + 1e-8  # Add a small value to avoid division by zero
    normalized_returns = (returns - mean) / std
    return normalized_returns

# Main training loop

distance_matrix = np.array([
    [0, 1/10, 1/8, 1/6, 1/4],
    [1/10, 0, 1/12, 1/9, 1/5],
    [1/8, 1/12, 0, 1/10, 1/6],
    [1/6, 1/9, 1/10, 0, 1/9],
    [1/4, 1/5, 1/6, 1/9, 0]
], dtype=float)/10


start_time = time.time()

env = ReentrantNetworkEnv(distance_matrix)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
model = ActorCritic(state_size, action_size)
optimizer_actor = tf.keras.optimizers.Adam(learning_rate=lr_actor)
optimizer_critic = tf.keras.optimizers.Adam(learning_rate=lr_critic)

max_episodes = 200
max_steps_per_episode = 1000
average_rewards_per_episode = []
packages_left_per_episode = []
total_failures = 0

# Main training loop
for episode in range(max_episodes):
    states, actions, rewards, values = [], [], [], []
    state = env.reset()
    done = False

    for step in range(max_steps_per_episode):
        state_tensor = tf.convert_to_tensor(state, dtype = tf.float32)
        state_tensor = tf.expand_dims(state_tensor, 0)
        logits, value = model(state_tensor)
        action = tf.random.categorical(logits, 1)[0, 0].numpy()

        next_state, reward, done, _ = env.step(action)

        states.append(state_tensor)
        actions.append(action)
        rewards.append(reward)
        values.append(value[0,0])

        state = next_state

        if done:
            break

    # Calculate discounted rewards
    rewards = discount_rewards(rewards, gamma)
    average_rewards_per_episode.append(rewards[0] / len(rewards))
    normalized_returns = normalize_returns(rewards)

    # Convert to tensors
    returns_batch = tf.convert_to_tensor(normalized_returns, dtype=tf.float32)
    states = tf.concat(states, axis=0)
    actions = np.array(actions, dtype=np.int32)
    values = tf.convert_to_tensor(np.array(values), dtype=tf.float32)
    old_logits, _ = model(states)

    # Update Actor-Critic Model
    policy_loss, value_loss = ppo_loss(
        model=model,
        optimizer_actor=optimizer_actor,
        optimizer_critic=optimizer_critic,
        old_logits=old_logits,
        old_values=values,
        advantages= returns_batch - values,
        states=states,
        actions=actions,
        returns=returns_batch
    )
    packages_left_per_episode.append(env.packages_left)
    total_failures += sum(env.failures)

end_time = time.time()
training_duration = end_time - start_time

average_failures = total_failures / max_episodes
print(f"Average number of failures: {average_failures}")

fig, axes = plt.subplots(1,2)
fig.suptitle('Medium Traffic')
fig.tight_layout(pad=3.0)
axes[0].plot(average_rewards_per_episode)
axes[0].set_xlabel('Episodes')
axes[0].set_ylabel('Average Amount of Packages')
#axes[0].axhline(y=50, color='purple', linestyle=':', linewidth=2, label=f'Buffer Capacity 1 (50)') # uncomment for restrictive buffer capacity
axes[1].plot(packages_left_per_episode)
axes[1].set_xlabel('Episodes')
axes[1].set_ylabel('Total Number of Packages Left')
plt.show()



print(f"Training completed in {training_duration:.2f} seconds")