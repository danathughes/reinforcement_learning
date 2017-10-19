## NFQ.py			Dana Hughes					17-Oct-2017
##
## 

import numpy as np
import random

import tensorflow as tf

class Q_Net:
	"""
	"""

	def __init__(self, state_size, action_size, num_hidden):
		"""
		"""

		# Create a session
		self.sess = tf.InteractiveSession()

		self.state = tf.placeholder(tf.float32, (None, state_size))
		self.action = tf.placeholder(tf.float32, (None, action_size))

		# State-to-hidden and action-to-hidden
		W_sh = tf.Variable(tf.truncated_normal((state_size, num_hidden), stddev=0.1))
		W_ah = tf.Variable(tf.truncated_normal((action_size, num_hidden), stddev=0.1))
		b_h = tf.Variable(tf.constant(0.1, shape=(num_hidden,)))

		# Hidden layer
		hidden = tf.sigmoid(tf.matmul(self.state, W_sh) + tf.matmul(self.action, W_ah) + b_h)

		# Hidden-to-output
		W_ho = tf.Variable(tf.truncated_normal((num_hidden, 1), stddev=0.1))
		b_o = tf.Variable(tf.constant(0.1, shape=(1,)))

		# Q Value
		self.Q = tf.matmul(hidden, W_ho) + b_o

		# Training
		self.target = tf.placeholder(tf.float32, (None,1))
		loss = 0.5 * tf.square(self.Q - self.target)

		self.train_step = tf.train.RMSPropOptimizer(0.1).minimize(loss)

		# Initialize the variables
		self.sess.run(tf.global_variables_initializer())
		



class NFQAgent:
	"""
	An agent which performs value iteration
	"""

	def __init__(self, environment, **kwargs):
		"""
		Create a new value iteration agent
		"""

		num_hidden = kwargs.get('num_hidden', 20)

		# Store the provided environment
		self.environment = environment

		# What are the possible states and actions?
		self.states = self.environment.states
		self.actions = self.environment.actions

		# What is the discount factor be?
		self.discount = kwargs.get('discount', 0.9)

		# What is the current state?
		self.state = None
		self.action = None
		self.reward = None
		self.experiences = []

		# Make the network
		self.net = Q_Net(np.sum(self.environment.shape), len(self.actions), num_hidden)


	def state_to_one_hot(self, state):
		"""
		Map Gridworld states to one-hot
		"""

		one_hot = np.zeros((np.sum(self.environment.shape),))
		one_hot[state[0]] = 1.0
		one_hot[self.environment.shape[0] + state[1]] = 1.0

		return one_hot


	def actions_to_one_hot(self, action):
		"""
		"""

		one_hot = np.zeros((4,))

		if action=='left':
			one_hot[0] = 1.0
		if action=='right':
			one_hot[1] = 1.0
		if action=='up':
			one_hot[2] = 1.0
		if action=='down':
			one_hot[3] = 1.0

		return one_hot



	def observe(self, state, is_terminal = False):
		"""
		Set the agent's state to the provided state
		"""

		# Create a new transition experience (if applicable)
		if self.state:
			experience = (self.state, action, reward, state, is_terminal)
			self.experiences.append(experience)

		self.state = state


	def act(self):
		"""
		Determine which action results in the best next state
		"""

		# Calculate the expected value after each action
		next_values = {}

		for action in self.actions:

			# Get the possible next states for performing this action in the current state
			next_states, probs = self.environment.transitions(self.state, action)

			# Add up the values of the next states
			value = 0.0
			for ns, p in zip(next_states, probs):
				value += p * self.environment.reward(self.state, action, ns) 
				value += p * self.discount * self.values[ns]

			next_values[action] = value

		# Determine the best action
		best_value = max(next_values.values())
		best_action = None

		for action in self.actions:
			if next_values[action] == best_value:
				best_action = action

		return best_action


	def reward(self, reward):
		"""
		Receive a reward from the environment
		"""

		self.reward = reward


	def reset(self):
		"""
		"""

		self.state = None
		self.experiences = []


	def best_Q(self, state):
		"""
		"""

		return 0


	def create_dataset(self):
		"""
		"""

		states = np.array([exp[0] for exp in self.experiences])
		actions = np.array([exp[1] for exp in self.experiences])

		inputs = (states, actions)

		targets = []
		for exp in self.experiences:
			if exp[4]:                  # Is it terminal?
				targets.append(exp[3])
			else:
				# What's the best action for the next state?
				Q = self.best_Q(state)
				targets.append(exp[2] + self.discount * Q)

		targets = np.array(targets)
