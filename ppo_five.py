import tensorflow as tf
import numpy as np
import gymnasium
from gymnasium import spaces
import matplotlib.pyplot as plt
import random

#seed = random.randint(0, 2**32 - 1)

seed = 4223724310
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

print('Verwendeter Seed: ', seed)

# Hyperparameters
lr_actor = 0.0001
lr_critic = 0.001
clip_ratio = 0.2
epochs = 10
batch_size = 32
gamma = 1.0
initial_entropy_coefficient = 0.8
scaling_factor = 1.0

# Parameters
max_iterations = 400
max_steps_per_iteration = 2000
buffer_capacity_1 = 50
buffer_capacity = 10
num_buffers = 5
arrival_rate = 0.1
arrival_rate_virtual = 0.05

distance_matrix = np.array([
    [0, 1.25, 1.8, 2.5, 3.2],
    [1.25, 0, 0.8, 1.6, 2.5],
    [1.8, 0.8, 0, 1, 1.8],
    [2.5, 1.6, 1, 0, 1.1],
    [3.2, 2.5, 1.8, 1.1, 0]
], dtype=float)

# Job Scheduling Environment
class ReentrantNetworkEnv(gymnasium.Env):
    def __init__(self, distance_matrix, arrival_rate, arrival_rate_virtual, buffer_capacity_1 = 500, buffer_capacity=10,
                 num_buffers=5,  failure_prob=0.05, max_steps_per_iteration = 1000):
        ''' Initialize the Reentrant Network Environment '''
        super(ReentrantNetworkEnv, self).__init__()

        self.distance_matrix = distance_matrix
        self.num_buffers = num_buffers
        self.buffer_capacity_1 = buffer_capacity_1
        self.buffer_capacity = buffer_capacity
        self.failure_prob = failure_prob
        self.failures = np.zeros(num_buffers)
        self.arrival_rate = arrival_rate
        self.arrival_rate_virtual = arrival_rate_virtual
        self.max_steps_per_iteration = max_steps_per_iteration

        self.state_size = 2 * num_buffers
        self.action_size = num_buffers + 1 # action for each buffer and one for wait action

        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.state_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.action_size)

        self.trigger_buffer = []
        self.packages_left = 0

        self.service_times_machines = [0.0]*num_buffers
        self.transportation_times_robot = 0.0
        self.waiting = 0

        self.reset()

    def reset(self):
        ''' Reset the environment to the initial state '''
        self.buffer = [[] for _ in range(self.num_buffers)]
        self.machines_busy = [[] for _ in range(self.num_buffers)]
        self.robot_busy = 0.0
        self.robot_distances = self.distance_matrix[0]
        self.failures = np.zeros(self.num_buffers)
        self.time_step = 0
        self.done = False
        self.trigger_buffer.clear()
        self.packages_left = 0

        self.service_times_machines = [0.0]*self.num_buffers
        self.transportation_times_robot = 0.0
        self.waiting = 0
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

        if self.robot_busy <= self.time_step:

            if 0 < action < self.num_buffers: # if any action but last buffer is chosen
                if 0 < len(self.buffer[action - 1]) and len(self.buffer[action]) < self.buffer_capacity and len(self.machines_busy[action-1]) < 1: # if current buffer is not empty and new buffer not full and machine not busy
                    self.buffer[action - 1].pop(0) # remove package from current buffer (FIFO)
                    service_return = self._service_time(buffer_index = action - 1) # call service time function
                    service_time = service_return[0]
                    ready_time = self.time_step + service_time # calculate ready time of package
                    self.machines_busy[action - 1].append((ready_time, service_return[1])) # add package to machine

                    pure_service_time = service_return[
                        2]  # get pure service time of current machine (i.e. without transportation)
                    self.service_times_machines[
                        action - 1] += pure_service_time  # track service times of current machine
                    self.transportation_times_robot += self.robot_distances[action - 1]  # track transportation time of robot

            elif action == self.num_buffers:  # if action is at last buffer
                if 0 < len(self.buffer[action - 1]) and len(self.machines_busy[action-1]) < 1: # if current buffer is not empty and last machine not busy
                    self.buffer[action - 1].pop(0)  # remove package from last buffer (FIFO)
                    service_return = self._service_time(buffer_index=action - 1)  # call service time function
                    service_time = service_return[0]
                    ready_time = self.time_step + service_time  # calculate ready time of package
                    self.machines_busy[action - 1].append((ready_time, service_return[1]))  # add package to machine

                    pure_service_time = service_return[
                        2]  # get pure service time of current machine (i.e. without transportation)
                    self.service_times_machines[
                        action - 1] += pure_service_time  # track service times of current machine
                    self.transportation_times_robot += self.robot_distances[
                        action - 1]  # track transportation time of robot

            if 0 < action <= self.num_buffers:
                self.robot_distances = self.distance_matrix[action - 1]  # set new location of robot at current buffer


        # if action is 0, do nothing
        if action == 0:
            self.waiting += 1

        total_packages = sum(len(b) for b in self.buffer)
        cost += total_packages

        self._package_arrivals()
        self._trigger_decision_epoch()

        if self.time_step >= self.max_steps_per_iteration:
            self.done = True

        next_state = self.get_state()
        return next_state, cost, self.done, {}

    def _package_arrivals(self):
        ''' Simulate actual and virtual package arrivals to the system '''
        new_packages = np.random.poisson(self.arrival_rate) # arrival rate of packages
        for _ in range(new_packages):
            if len(self.buffer[0]) < self.buffer_capacity_1: # if first buffer is not full
                self.buffer[0].append(self.time_step) # add package to first buffer (ready immediately)

        virtual_arrival = np.random.poisson(self.arrival_rate_virtual) # simulate virtual arrivals to wait buffer
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
        scale_param = 3  # scale parameter
        stochastic_time = np.random.weibull(shape_param) * scale_param

        self.robot_busy = self.time_step + self.robot_distances[buffer_index]

        service_time = float(
            stochastic_time + self.robot_distances[
                buffer_index])  # Total Service time is processing time plus transportation time

        if np.random.rand() < self.failure_prob:  # Simulate failure with probability failure_prob
            self.failures[buffer_index] += 1  # Track failures
            service_time = 2 * service_time  # double service time as penalty
            success = False
        else:
            self.failures[buffer_index] = 0
            success = True

        return service_time, success, stochastic_time

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
        self.actor_output = tf.keras.layers.Dense(action_size, activation=None) # Policy logits
        self.critic_output = tf.keras.layers.Dense(1)  # Value output


    def call(self, state):
        actor_x = self.actor_layer(state)
        critic_x = self.critic_layer(state)
        logits = self.actor_output(actor_x)  # Actor output (policy logits)
        value = self.critic_output(critic_x)  # Critic output (value estimate)
        return logits, value


# PPO Loss Function

def ppo_loss(model, optimizer_actor, optimizer_critic, old_logits, old_values,
             advantages, states, actions, returns, entropy_coefficient):
    action_size = model.actor_output.units

    def compute_loss(logits, values, actions, returns, entropy_coefficient):
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
        entropy_bonus = - tf.reduce_mean(policy * tf.math.log(policy + 1e-10))
        entropy = -tf.reduce_mean(tf.reduce_sum(policy * tf.math.log(policy + 1e-10), axis=1))

        total_loss = policy_loss + 0.5 * value_loss - entropy_coefficient * entropy_bonus
        return total_loss, policy_loss, value_loss, entropy

    @tf.function
    def train_step(states, actions, returns, old_logits, old_values,
                   advantages, entropy_coefficient):
        with tf.GradientTape() as tape:
            logits, values = model(states, training=True)

            # Compute losses
            total_loss, policy_loss, value_loss, entropy = compute_loss(logits, values, actions, returns, entropy_coefficient)

        # Compute and apply gradients
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer_actor.apply_gradients(zip(gradients, model.trainable_variables))
        optimizer_critic.apply_gradients(zip(gradients, model.trainable_variables))

        return policy_loss, value_loss, entropy

    for _ in range(epochs):
        policy_loss, value_loss, entropy = train_step(states, actions, returns, old_logits, old_values, advantages, entropy_coefficient)

    return policy_loss, value_loss, entropy


def normalize_returns(returns):
    mean = np.mean(returns)
    std = np.std(returns) + 1e-8  # Add a small value to avoid division by zero
    normalized_returns = (returns - mean) / std
    return normalized_returns

# Adaptive Entropy Coefficient
def adaptive_entropy_coefficient(entropy, target_entropy, initial_coefficient,
                                 scaling_factor):
    if entropy > target_entropy:
        return initial_coefficient/scaling_factor
    else:
        return initial_coefficient*scaling_factor


# Main training loop

env = ReentrantNetworkEnv(distance_matrix = distance_matrix, arrival_rate = arrival_rate, arrival_rate_virtual = arrival_rate_virtual,
                          buffer_capacity_1 = buffer_capacity_1, buffer_capacity = buffer_capacity,
                          num_buffers = num_buffers, max_steps_per_iteration = max_steps_per_iteration)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
model = ActorCritic(state_size, action_size)
optimizer_actor = tf.keras.optimizers.Adam(learning_rate=lr_actor)
optimizer_critic = tf.keras.optimizers.Adam(learning_rate=lr_critic)

# Initializing
average_rewards_per_iteration = []
packages_left_per_iteration = []
entropy_per_iteration = []
policy_loss_per_iteration = []
failures_per_iteration = []
target_entropy = 0 #-np.log(1.0 / action_size) * 0.98
saved_logits = None
machine_loads = [[] for _ in range(num_buffers)]
robot_load = []
waiting = []
first_iteration_buffer_counts = []
middle_iteration_buffer_counts = []
last_iteration_buffer_counts = []

# Main training loop
for iteration in range(max_iterations):
    states, actions, rewards, values = [], [], [], []
    returns_batch = []
    state = env.reset()
    done = False

    buffer_counts = []

    for step in range(max_steps_per_iteration):
        state_tensor = tf.convert_to_tensor(state, dtype = tf.float32)
        state_tensor = tf.expand_dims(state_tensor, 0)
        logits, value = model(state_tensor)
        action = tf.random.categorical(logits, 1)[0, 0].numpy()

        if iteration == 0 and step == 0: # initial entropy value
            policy = tf.nn.softmax(logits)
            initial_entropy_value = tf.reduce_sum(-policy * tf.math.log(policy + 1e-10))
            entropy_per_iteration.append(initial_entropy_value)

        next_state, reward, done, _ = env.step(action)

        states.append(state_tensor)
        actions.append(action)
        rewards.append(reward)
        values.append(value[0,0])

        state = next_state

        buffer_counts.append(len(env.buffer[0]))

        if done:
            break

    # Save buffer counts for first, middle and last iteration
    if iteration == 0:
        first_iteration_buffer_counts = buffer_counts
    elif iteration == max_iterations // 2 - 1:
        middle_iteration_buffer_counts = buffer_counts
    elif iteration == max_iterations - 1:
        last_iteration_buffer_counts = buffer_counts

    # Calculate returns batch
    discounted_sum = 0
    for r in rewards[::-1]:
        discounted_sum = r + gamma * discounted_sum
        returns_batch.append(discounted_sum)
    returns_batch.reverse()
    returns_batch = normalize_returns(returns_batch)
    # Convert to tensors
    returns_batch = tf.convert_to_tensor(returns_batch, dtype=tf.float32)
    states = tf.concat(states, axis=0)
    actions = np.array(actions, dtype=np.int32)
    values = tf.convert_to_tensor(np.array(values), dtype=tf.float32)
    old_logits, _ = model(states)

    saved_logits = logits  # Save logits for model evaluation

    # Update Actor-Critic Model
    policy_loss, value_loss, entropy = ppo_loss(
        model=model,
        optimizer_actor=optimizer_actor,
        optimizer_critic=optimizer_critic,
        old_logits=old_logits,
        old_values=values,
        advantages= returns_batch - values,
        states=states,
        actions=actions,
        returns=returns_batch,
        entropy_coefficient = initial_entropy_coefficient
    )

    # Calculate average rewards
    average_rewards = sum(rewards) / len(rewards)
    average_rewards_per_iteration.append(average_rewards)
    # Calculate total throughput
    packages_left_per_iteration.append(env.packages_left)
    # Calculate entropy
    entropy_per_iteration.append(entropy.numpy())
    # Adapt entropy coefficient
    initial_entropy_coefficient = adaptive_entropy_coefficient(entropy.numpy(), target_entropy,
                                                               initial_entropy_coefficient, scaling_factor)
    # Track policy loss
    policy_loss_per_iteration.append(policy_loss.numpy())

    # Track machine loads
    for i in range(num_buffers):
        if env.service_times_machines[i] > 0:
            machine_load = env.service_times_machines[i]/max_steps_per_iteration
            machine_loads[i].append(machine_load)
        else:
            machine_loads[i].append(0)

    # Track Gantry Robot load
    if env.transportation_times_robot > 0.0:
        robot_load.append(env.transportation_times_robot/max_steps_per_iteration)

    # Track waiting time
    waiting.append(env.waiting/max_steps_per_iteration)

    # Track failures
    failures_per_iteration.append(sum(env.failures))

# Plotting
# Plot average rewards and total throughput

# determine which type of traffic regime is used
if arrival_rate > 0.1:
    regime = 'High Traffic Scenario'
elif arrival_rate > 0.01:
    regime = 'Medium Traffic Scenario'
else:
    regime = 'Low Traffic Scenario'
fig, axes = plt.subplots(1,2)
fig.suptitle(f'{regime}')
fig.tight_layout(pad=3.0)
axes[0].plot(average_rewards_per_iteration)
axes[0].set_xlabel('Policy Iterations')
axes[0].set_ylabel('Average Amount of Packages')
axes[0].axhline(y=50, color='purple', linestyle=':', linewidth=2, label=f'Buffer Capacity 1 (50)') # uncomment for restrictive buffer capacity
axes[1].plot(packages_left_per_iteration)
axes[1].set_xlabel('Policy Iterations')
axes[1].set_ylabel('Total Throughput')
plt.show()
# Plot policy loss and entropy
fig, axes = plt.subplots(1,2)
fig.suptitle('Policy Loss and Entropy')
fig.tight_layout(pad=3.0)
axes[0].plot(policy_loss_per_iteration)
axes[0].set_xlabel('Policy Iterations')
axes[0].set_ylabel('Policy Loss')
axes[1].plot(entropy_per_iteration)
axes[1].set_xlabel('Policy Iterations')
axes[1].set_ylabel('Entropy')
plt.show()

# Plot machine loads
plt.figure(figsize=(10, 6))
for i in range(num_buffers):
    plt.plot(machine_loads[i], label=f'Machine {i+1} Load')
plt.plot(robot_load, label='Gantry Robot Load', linestyle = '--')
plt.plot(waiting, label = 'Action 0', linestyle = '-.')
plt.xlabel('Policy Iterations')
plt.ylabel('Load')
#plt.title('Machine Loads Over Policy Iterations')
plt.legend()
plt.show()

# Plot the counts of packages in the first buffer for the first, 100th, and last iteration
plt.figure(figsize=(10, 6))
plt.plot(first_iteration_buffer_counts, label='First Iteration')
plt.plot(middle_iteration_buffer_counts, label=f'{(max_iterations // 2)}th Iteration')
plt.plot(last_iteration_buffer_counts, label='Last Iteration')
plt.xlabel('Time Steps')
plt.ylabel('Count of Packages in First Buffer')
#plt.title('Count of Packages in First Buffer Over Time')
plt.legend()
plt.show()

# Plot the failure counts per iteration
plt.figure(figsize=(10, 6))
plt.plot(failures_per_iteration)
plt.xlabel('Policy Iterations')
plt.ylabel('Number of Failures')
plt.title('Failures Over Policy Iterations')
plt.legend()
plt.show()



