## value_iteration_agent.py			Dana Hughes				09-Oct-2017
 ##
## Agents which perform value iteration (i.e., learns V(s) or Q(s,a))
## in order to learn a policy


import numpy as np
import random

class ValueIterationAgent:
	"""
	An agent which performs value iteration
	"""

	def __init__(self, environment, **kwargs):
		"""
		Create a new value iteration agent
		"""

		# Store the provided environment
		self.environment = environment

		# What are the possible states and actions?
		self.states = self.environment.states
		self.actions = self.environment.actions

		# The value function will be stored in a table mapping
		# states (from the environment) to values
		self.values = {}
		self.initialize_values()

		# What is the discount factor be?
		self.discount = kwargs.get('discount', 0.9)

		# What is the current state?
		self.state = None


	def initialize_values(self, std_dev = 0.0):
		"""
		Set up the value table.  Values are initialized using
		a normal distribution centered around 0.

		std_dev - standard deviation of the normal distribution
		          0.0 sets all values to 0.0
		"""

		for state in self.states:
			self.values[state] = np.random.normal(0, std_dev)


	def observe(self, state):
		"""
		Set the agent's state to the provided state
		"""

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


	def update_values(self):
		"""
		Perform a single value update
		"""

		max_difference = 0.0

		# Place to store new state values
		new_values = {}

		# Go through each state and update its value
		for state in self.states:
			# Calculate the value of each action
			action_values = []

			for action in self.actions:
				# What is the transition function?
				next_states, probs = self.environment.transitions(state, action)

				# Calculate the future reward (reward + discounted next state value) for 
				# each of the next states, weighted by the transition probability
				future_reward = 0.0

				for ns, p in zip(next_states, probs):
					future_reward += p * (self.environment.reward(state, action, ns))    # Immediate reward
					future_reward += p * self.discount * self.values[ns]                 # Discounted future value

				action_values.append(future_reward)

			# What is the value of the best action in this state?
			new_values[state] = max(action_values)

			# Is the state terminal?  Then the values is simply 0.
			# Reward is applied for *entering* the state
			if state in self.environment.is_terminal:
				new_values[state] = 0.0

			# Update the maximum difference between value updates
			max_difference = max(max_difference, abs(new_values[state] - self.values[state]))

		# All the state's new values are calculated.  Update the agent's value function
		self.values = new_values

		return max_difference


	def learn(self, threshold=1e-4):
		"""
		Update the value function of all states until convergence 
		(largest value improvement less than threshold)
		"""

		# Ensure at least one iteration
		max_difference = 2*threshold

		while max_difference > threshold:
			max_difference = self.update_values()


class QValueIterationAgent:
	"""
	An agent which performs Q-value iteration
	"""

	def __init__(self, environment, **kwargs):
		"""
		Create a new value iteration agent
		"""

		# Store the provided environment
		self.environment = environment

		# What are the possible states and actions?
		self.states = self.environment.states
		self.actions = self.environment.actions

		# The value function will be stored in a table mapping
		# states (from the environment) to values
		self.values = {}
		self.initialize_values()

		# What is the discount factor be?
		self.discount = kwargs.get('discount', 0.9)

		# What is the current state?
		self.state = None


	def initialize_values(self, std_dev = 0.0):
		"""
		Set up the value table.  Values are initialized using
		a normal distribution centered around 0.

		std_dev - standard deviation of the normal distribution
		          0.0 sets all values to 0.0
		"""

		for state in self.states:
			for action in self.actions:
				self.values[(state, action)] = np.random.normal(0, std_dev)


	def observe(self, state):
		"""
		Set the agent's state to the provided state
		"""

		self.state = state


	def act(self):
		"""
		Determine which action results in the best next state
		"""

		# Calculate the Q value for each action
		Q = {}

		# Go through each action and determine the Q value, and see if there's improvement
		for action in self.actions:
			Q[action] = self.values[(self.state, action)]

		# Determine the best action
		best_Q = max(Q.values())
		best_action = None

		for action in self.actions:
			if Q[action] == best_Q:
				best_action = action

		return best_action


	def learn(self, threshold=1e-4):
		"""
		Update the value function of all states until convergence 
		(largest value improvement less than threshold)
		"""

		# Ensure at least one iteration
		max_difference = 2*threshold

		while max_difference > threshold:

			max_difference = 0.0

			# Place to store new state values
			new_values = {}

			# Go through each state / action pair and update its value
			for state in self.states:
				for action in self.actions:

					# What is the transition function?
					next_states, probs = self.environment.transitions(state, action)

					# Calculate the future reward (reward + discounted next state value) for 
					# each of the next states, weighted by the transition probability
					action_value = 0.0

					for ns, p in zip(next_states, probs):
						action_value += p * (self.environment.reward(state, action, ns))    # Immediate reward

						best_next_action = max([self.values[(ns, a)] for a in self.actions])
						action_value += p * self.discount * best_next_action                # Discounted future value

					# Update the Q value for this action-value pair
					new_values[(state, action)] = action_value

					# Is the state terminal?  Then the values is simply 0.
					# Reward is applied for *entering* the state
					if state in self.environment.is_terminal:
						new_values[(state, action)] = 0.0

					# Update the maximum difference between old and new values
					max_difference = max(max_difference, abs(new_values[(state,action)] - self.values[(state,action)]))


			# All the state's new values are calculated.  Update the agent's value function
			self.values = new_values