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

		self.train_step = tf.train.RMSPropOptimizer(0.05).minimize(loss)

		# Initialize the variables
		self.sess.run(tf.global_variables_initializer())


	def get_Q(self, state, action):
		"""
		"""

		state = np.reshape(state, (1,) + state.shape)
		action = np.reshape(action, (1,) + action.shape)

		return self.sess.run(self.Q, feed_dict = {self.state: state, self.action: action})[0,0]


	def train(self, states, actions, targets):
		"""
		"""

		fd = {self.state: states, self.action: actions, self.target: targets}

		self.sess.run(self.train_step, feed_dict = fd)
		



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

		# Epsilon-greedy value
		self.epsilon = kwargs.get('epsilon', 1.0)

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


	def action_to_one_hot(self, action):
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
			experience = (self.state, self.action, self.reward, state, is_terminal)
			self.experiences.append(experience)

		self.state = state


	def act(self):
		"""
		Determine which action results in the best next state
		"""

		if random.random() < self.epsilon:
			action = random.choice(list(self.actions))
			self.action = action
			return action

		state = self.state_to_one_hot(self.state)

		best_action = list(self.actions)[0]
		best_Q = self.net.get_Q(state, self.action_to_one_hot(best_action))

		for action in self.actions:
			if self.net.get_Q(state, self.action_to_one_hot(action)) > best_Q:
				best_action = action

		self.action = action

		return best_action


	def give_reward(self, reward):
		"""
		Receive a reward from the environment
		"""

		self.reward = reward


	def reset(self):
		"""
		"""

		self.state = None
		self.action = None
		self.reward = None
		self.experiences = []


	def best_Q(self, state):
		"""
		"""

		state_one_hot = self.state_to_one_hot(state)

		best_Q = self.net.get_Q(state_one_hot, self.action_to_one_hot(list(self.actions)[0]))

		for action in self.actions:
			best_Q = max(best_Q, self.net.get_Q(state_one_hot, self.action_to_one_hot(action)))

		return best_Q


	def create_dataset(self):
		"""
		"""

		states = []
		actions = []

#		states = np.array([exp[0] for exp in self.experiences])
#		actions = np.array([exp[1] for exp in self.experiences])

#		inputs = (states, actions)

		targets = []
		for exp in self.experiences:
			states.append(self.state_to_one_hot(exp[0]))
			actions.append(self.action_to_one_hot(exp[1]))

			if exp[4]:                  # Is it terminal?
				targets.append(exp[3])
			else:
				# What's the best action for the next state?
				Q = self.best_Q(exp[0])
				targets.append(exp[2] + self.discount * Q)

		states = np.array(states)
		actions = np.array(actions)
		targets = np.array(targets)

		return states, actions, targets


	def learn(self, train_steps = 1):
		states, actions, targets = self.create_dataset()

		targets = np.reshape(targets, targets.shape + (1,))

		for i in range(train_steps):
			self.net.train(states, actions, targets)