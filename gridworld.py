## gridworld.py			Dana Hughes				09-Oct-2017
##
## An gridworld environment

import numpy as np
import random

class GridWorld:
	"""
	A Gridworld Environment
	"""

	def __init__(self, width, height, **kwargs):
		"""
		Create a new gridworld
		"""

		self.shape = (width, height)

		# Where does an agent start when reset, and what is the
		# agent's current state?
		self.start_cell = kwargs.get('start_cell', (0,0))
		self.state = self.start_cell

		# What are the terminal states, blocked cells, and rewards?
		self.is_terminal = np.zeros(self.shape).astype(np.bool)
		self.is_blocked = np.zeros(self.shape).astype(np.bool)
		self.rewards = np.zeros(self.shape)

		# A list of possible states and actions
		self.states = None 									# Will be filled in as needed
		self.actions = range(4)								# enumeration of cardinal directions
		self.action_names = ['left','right','up','down']

		# What is the probability of performing a random action?
		self.noise = kwargs.get('noise', 0.1)

		# Reward for action, and for collision
		self.action_reward = kwargs.get('action_reward', -1.0)
		self.collision_reward = kwargs.get('collision_reward', -10.0)


	def get_state(self):
		"""
		What state is the agent currently in?
		"""

		return self.state


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

		# If the state is terminal, then simply remain in this state
		# with probability 1.0
		if self.is_terminal[state]:
			return [state], [1.0]

		# Unpack the current state
		x,y = state

		# Assume all neighboring states can be transitioned into
		next_states = [(x-1,y), (x+1,y), (x,y+1), (x,y-1), (x,y)]
		probabilities = [0.0, 0.0, 0.0, 0.0, 0.0]

		# Calculate the probabilites of each transition
		# Each action should have a baseline probability of noise / 4,
		# and the selected action should have an additional probability
		# of (1.0 - noise)

		base_prob = self.noise / 4

		# Can move left?
		if x > 0 and not self.is_blocked[x-1,y]:
			probabilities[0] += base_prob + (action == 0)*(1.0 - self.noise)
		else:
			probabilities[4] += base_prob + (action == 0)*(1.0 - self.noise)			# Bumped into something -- stay put

		# Right?
		if x < self.shape[0] - 1 and not self.is_blocked[x+1,y]:
			probabilities[1] += base_prob + (action == 1)*(1.0 - self.noise)
		else:
			probabilities[4] += base_prob + (action == 1)*(1.0 - self.noise)

		# Up?
		if y < self.shape[1] - 1 and not self.is_blocked[x,y+1]:
			probabilities[2] += base_prob + (action == 2)*(1.0 - self.noise)
		else:
			probabilities[4] += base_prob + (action == 2)*(1.0 - self.noise)

		# Down?
		if y > 0 and not self.is_blocked[x,y-1]:
			probabilities[3] += base_prob + (action == 3)*(1.0 - self.noise)
		else:
			probabilities[4] += base_prob + (action == 3)*(1.0 - self.noise)


		# Return all non-zero transition probabilities
		states = []
		probs = []

		for s,p in zip(next_states, probabilities):
			if p > 0:
				states.append(s)
				probs.append(p)

		return states, probs


	def reward(self, state, action, next_state):
		"""
		Calculate the reward of performing the action in the state, and 
		transitioning to the next state
		"""

		reward = 0.0

		# What is the reward for entering next_state?
		reward += self.rewards[next_state]

		# What is the reward for performing an action?
		reward += self.action_reward

		# Did the agent bump into a wall?
		if state == next_state:
			reward += self.collision_reward

		return reward


	def block(self, x, y):
		"""
		Block cell (x,y)
		"""

		assert x < self.shape[0] and x > 0, "x not a valid point in grid"
		assert y < self.shape[1] and y > 0, "y not a valid point in grid"

		self.is_blocked[x,y] = True


	def add_terminal(self, x, y):
		"""
		Make cell (x,y) a terminal state
		"""

		assert x < self.shape[0] and x > 0, "x not a valid point in grid"
		assert y < self.shape[1] and y > 0, "y not a valid point in grid"

		self.is_terminal[x,y] = True


	def set_reward(self, x, y, reward):
		"""
		Add a reward for entering cell (x,y)
		"""

		assert x < self.shape[0] and x > 0, "x not a valid point in grid"
		assert y < self.shape[1] and y > 0, "y not a valid point in grid"

		self.rewards[x,y] = reward


	def get_states(self):
		"""
		Return a list of all possible states
		"""

		# Create the set of states if they don't already exist
		if not self.states:
			self.states = []
			cells = [(i,j) for i in range(self.shape[0]) for j in range(self.shape[1])]

			for cell in cells:
				if not self.is_blocked[cell]:
					self.states.append(cell)

		return self.states


	def get_actions(self):
		"""
		Return a list of all possible actions
		"""

		return self.actions


	def action_name(self, action_num):
		"""
		Descriptive name of the action
		"""

		return self.action_names[action_num]


