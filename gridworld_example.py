from environments.gridworld import *
from agents.value_iteration_agent import *

from environments.visualizer import *

import random
from time import sleep

WIDTH = 25
HEIGHT = 30


def make_random_gridworld(width=WIDTH, height=HEIGHT, **kwargs):
	"""
	Create a random gridworld environment
	"""

	# Get optional arguments or default values
	noise = kwargs.get('noise', 0.1)
	pos_reward = kwargs.get('pos_reward', 100)
	neg_reward = kwargs.get('neg_reward', -100)
	num_blocks = kwargs.get('num_blocks', 250)
	reward_min_x = kwargs.get('reward_min_x', 5)
	reward_min_y = kwargs.get('reward_min_y', 5)
	num_pos_rewards = kwargs.get('num_pos_rewards', 3)
	num_neg_rewards = kwargs.get('num_neg_rewards', 3)

	# Make an environment and add a few random reward areas and blocks
	environment = GridWorld(25,30, noise=noise)

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


###
### Main Program
###


# Create an environment
environment = make_random_gridworld()
visualizer = make_visualizer(environment)
visualizer.draw()

# Create an agent
agent = ValueIterationAgent(environment)


# Get user input and perform various actions
inp = raw_input()

while inp != 'q':

	if inp == 'v':
		agent.update_values()
	elif inp == 's':
		agent.observe(environment.state)
		action = agent.act()
		print action
		environment.act(action)
	elif inp == 'r':
		environment.reset()
		environment.state = (random.randint(0, WIDTH-1), random.randint(0, HEIGHT-1))
	elif inp == 'l':
		agent.learn()
	elif inp == 't':
		while not environment.state in environment.is_terminal:
			agent.observe(environment.state)
			action = agent.act()
			print action
			environment.act(action)
			x,y = environment.state
			visualizer.set_agent_loc(x,y)
			visualizer.draw()
			sleep(0.5)



	for x in range(WIDTH):
		for y in range(HEIGHT):
			val = agent.values[(x,y)]
			visualizer.set_value(x,y,val)

	for loc in environment.is_terminal:
		x,y = loc
		reward = environment.rewards.get(loc, 0)
		visualizer.set_value(x,y,reward)

	x,y = environment.state
	visualizer.set_agent_loc(x,y)

	visualizer.draw()

	inp = raw_input()