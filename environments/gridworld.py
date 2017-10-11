## gridworld.py			Dana Hughes				09-Oct-2017
##
## An gridworld environment

import numpy as np
import random

from environment import DiscreteEnvironment

class GridWorld(DiscreteEnvironment):
	"""
	A Gridworld Environment
	"""

	def __init__(self, width, height, **kwargs):
		"""
		Create a new gridworld
		"""

		# Initialize a discrete environment
		DiscreteEnvironment.__init__(self, **kwargs)

		# Gridworld-specific attributes
		self.shape = (width, height)

		# Populate the states and actions
		for state in [(x,y) for x in range(width) for y in range(height)]:
			self.states.add(state)
		for action in ['up','down','left','right']:
			self.actions.add(action)

		# Where does an agent start when reset, and what is the
		# agent's current state?
		self.start_cell = kwargs.get('start_cell', (0,0))
		self.state = self.start_cell

		# What are the blocked cells?
		self.is_blocked = set()

		# Motion noise - probability of taking a random action
		self.noise = kwargs.get('noise', 0.1)

		# Reward for collision
		self.collision_reward = kwargs.get('collision_reward', -10.0)


	def transitions(self, state, action):
		"""
		For the given state and action, provide which states can be transitioned into,
		and the probability of transitioning into that state
		"""

		# If the state is terminal, then simply remain in this state
		# with probability 1.0
		if state in self.is_terminal:
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
		if x > 0 and not (x-1,y) in self.is_blocked:
			probabilities[0] += base_prob + (action == 'left')*(1.0 - self.noise)
		else:
			probabilities[4] += base_prob + (action == 'left')*(1.0 - self.noise)			# Bumped into something -- stay put

		# Right?
		if x < self.shape[0] - 1 and not (x+1,y) in self.is_blocked:
			probabilities[1] += base_prob + (action == 'right')*(1.0 - self.noise)
		else:
			probabilities[4] += base_prob + (action == 'right')*(1.0 - self.noise)

		# Up?
		if y < self.shape[1] - 1 and not (x,y+1) in self.is_blocked:
			probabilities[2] += base_prob + (action == 'up')*(1.0 - self.noise)
		else:
			probabilities[4] += base_prob + (action == 'up')*(1.0 - self.noise)

		# Down?
		if y > 0 and not (x,y-1) in self.is_blocked:
			probabilities[3] += base_prob + (action == 'down')*(1.0 - self.noise)
		else:
			probabilities[4] += base_prob + (action == 'down')*(1.0 - self.noise)


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

		reward = DiscreteEnvironment.reward(self, state, action, next_state)

		# Did the agent bump into a wall?
		if state == next_state:
			reward += self.collision_reward

		return reward


	def block(self, x, y):
		"""
		Block cell (x,y)
		"""

		assert (x,y) in self.states, "%s not a valid state!" % str((x,y))

		self.is_blocked.add((x,y))


	def reset(self):
		"""
		Set the position of the agent to the initial cell
		"""

		self.state = self.start_cell