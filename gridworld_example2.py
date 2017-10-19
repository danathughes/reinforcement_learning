from environments.gridworld import *
from agents.NFQ import *

from environments.visualizer import *

import random
from time import sleep

WIDTH = 20
HEIGHT = 10


def make_random_gridworld(width=WIDTH, height=HEIGHT, **kwargs):
	"""
	Create a random gridworld environment
	"""

	# Get optional arguments or default values
	noise = kwargs.get('noise', 0.1)
	pos_reward = kwargs.get('pos_reward', 100)
	neg_reward = kwargs.get('neg_reward', -100)
	num_blocks = kwargs.get('num_blocks', 50)
	reward_min_x = kwargs.get('reward_min_x', 5)
	reward_min_y = kwargs.get('reward_min_y', 5)
	num_pos_rewards = kwargs.get('num_pos_rewards', 3)
	num_neg_rewards = kwargs.get('num_neg_rewards', 3)

	# Make an environment and add a few random reward areas and blocks
	environment = GridWorld(width, height, noise=noise)

	# Add positive rewards, at least a few steps away from the start
	for i in range(num_pos_rewards):
		x = random.randint(reward_min_x, width-1)
		y = random.randint(reward_min_y, height-1)
		environment.set_reward((x,y),pos_reward)
		environment.add_terminal((x,y))

	for i in range(3):
		x = random.randint(reward_min_x, width-1)	
		y = random.randint(reward_min_y, height-1)
		environment.set_reward((x,y),neg_reward)
		environment.add_terminal((x,y))

	# Add some blocks
	for i in range(num_blocks):
		x = random.randint(1, width-1)
		y = random.randint(1, height-1)
		environment.block(x,y)

	return environment


def make_visualizer(environment):
	"""
	Create visualization for the environment\
	"""

	width, height = environment.shape

	# Create a display - add blocked areas and set terminal area rewards
	visualizer = GridworldVisualizer(width, height, 25, 25)
	for loc in environment.is_blocked:
		x,y = loc
		visualizer.set_blocked(x,y)
	for loc in environment.is_terminal:
		x,y = loc
		reward = environment.rewards.get(loc, 0)
		visualizer.set_value(x,y,reward)

	return visualizer


def run_episode(agent, environment, visualizer, max_steps=1000):
	"""
	Run an episode
	"""

	steps = 0

	environment.reset()

	while steps < max_steps:

		# 1. Get the state of the environment
		state = environment.state
		terminal = state in environment.is_terminal

		# Draw the agent
		if visualizer:
			x,y = state
			visualizer.set_agent_loc(x,y)
			visualizer.draw()

		# 1A. Is the state terminal?
		if terminal:
			steps = max_steps
		else:

			# 2. Agent observes the state
			agent.observe(state, terminal)

			# 3. Select an action
			action = agent.act()

			# 4. Perform the action, and get the reward
			reward = environment.act(action)

			# 5. Tell the agent about the reward
			agent.give_reward(reward)

			steps += 1


def update_visualizer(agent, environment, visualizer):
	"""
	"""

	for state in environment.states:
		x,y = state
		val = agent.best_Q(state)
		visualizer.set_value(x,y,val)

	for loc in environment.is_terminal:
		x,y = loc
		reward = environment.rewards.get(loc, 0)
		visualizer.set_value(x,y,reward)

	x,y = environment.state
	visualizer.set_agent_loc(x,y)

	visualizer.draw()



###
### Main Program
###


# Create an environment
environment = make_random_gridworld()
visualizer = make_visualizer(environment)
visualizer.draw()

# Create an agent
agent = NFQAgent(environment)


# Get user input and perform various actions
inp = raw_input()

while inp != 'q':

	if inp == 'r':
		run_episode(agent, environment, visualizer)
	elif inp == 'l':
		agent.learn()
	elif inp == 'u':
		update_visualizer(agent, environment, visualizer)
	elif inp == 't':
		run_episode(agent, environment, visualizer)
		print "Training...",
		agent.learn(20)
		print "Done"
		update_visualizer(agent, environment, visualizer)
	elif inp == '+e':
		agent.epsilon = min(agent.epsilon+0.1, 1.0)
		print agent.epsilon
	elif inp == '-e':
		agent.epsilon = max(agent.epsilon - 0.1, 0.0)
		print agent.epsilon

	inp = raw_input()