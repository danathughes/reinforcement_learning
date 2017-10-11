## environment.py			Dana Hughes				10-Oct-2017
##
## Abstract environment types

import numpy as np
import random

class DiscreteEnvironment:
	"""
	An environment with discrete states
	"""

	def __init__(self, **kwargs):
		"""
		Create a discrete environment
		"""

		# What are the terminal states, blocked cells, and rewards?
		self.is_terminal = set()
		self.rewards = {}

		# A list of possible states and actions
		self.states = set()								# Will be filled in as needed
		self.actions = set()

		# Common reward for performing an action
		self.action_reward = kwargs.get('action_reward', 0.0)


	def add_state(self, state, reward=0.0, terminal=False):
		"""
		Add a state to the environment
		"""

		assert not state in self.states, "State %s already exists in environment" % str(state)

		self.states.add(state)

		# Does this state have an associated reward?
		if reward != 0.0:
			self.rewards[state] = reward

		# Is this a terminal state?
		if terminal:
			self.is_terminal.add(state)


	def add_action(self, action):
		"""
		Add a new action to the environment
		"""

		assert not action in self.actions, "Action %s already defined in environment" % str(action)

		self.actions.add(action)


	def add_terminal(self, state):
		"""
		Make the state a terminal state
		"""

		assert state in self.states, "%s not an existing state in the environment!" % str(state)

		self.is_terminal.add(state)


	def set_reward(self, state, reward):
		"""
		Add a reward for entering the state
		"""

		assert state in self.states, "%s not an existing state in the environment!" % str(state)

		self.rewards[state] = reward


	def act(self, action):
		"""
		Perform some action, and return a reward
		"""

		# Get transition probabilities and next states given the action,
		next_states, probs = self.transitions(self.state, action)
		
		# Pick an action at (weighted) random
		rnd = random.random()
		next_state_idx = -1

		while rnd > 0:
			next_state_idx += 1
			rnd -= probs[next_state_idx]

		# Update the current state, and get the reward
		next_state = next_states[next_state_idx]
		reward = self.reward(self.state, action, next_state)

		self.state = next_state

		return reward


	def transitions(self, state, action):
		"""
		For the given state and action, provide which states can be transitioned into,
		and the probability of transitioning into that state
		"""

		# This is domain specific
		pass


	def reward(self, state, action, next_state):
		"""
		Calculate the reward of performing the action in the state, and 
		transitioning to the next state
		"""

		# The base reward is simply the reward for entering a state, plus any
		# reward for performing an arbitrary action
		# Subclasses may replace or extend the reward function

		reward = 0.0

		# What is the reward for entering next_state?
		reward += self.rewards.get(next_state, 0.0)

		# What is the reward for performing an action?
		reward += self.action_reward

		return reward

	def reset(self):
		"""
		Return the environment to some initial state
		"""

		# This is domain specific
		pass