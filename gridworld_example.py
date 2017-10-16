from environments.gridworld import *
from agents.value_iteration_agent import *

from environments.visualizer import *

import random
from time import sleep

WIDTH = 25
HEIGHT = 30

REWARD_MIN_X = 5
REWARD_MIN_Y = 5

# Make an environment and add a few random reward areas and blocks
environment = GridWorld(25,30, noise=0.1)

# Add three positive and negative rewards, at least a few steps away from the start
for i in range(3):
	x = random.randint(REWARD_MIN_X, WIDTH-1)
	y = random.randint(REWARD_MIN_Y, HEIGHT-1)
	environment.set_reward((x,y),100)
	environment.add_terminal((x,y))

for i in range(3):
	x = random.randint(REWARD_MIN_X, WIDTH-1)
	y = random.randint(REWARD_MIN_Y, HEIGHT-1)
	environment.set_reward((x,y),-100)
	environment.add_terminal((x,y))

# Add some blocks
for i in range(250):
	x = random.randint(1, WIDTH-1)
	y = random.randint(1, HEIGHT-1)
	environment.block(x,y)

# Create a display - add blocked areas and set terminal area rewards
visualizer = GridworldVisualizer(WIDTH, HEIGHT, 25, 25)
for loc in environment.is_blocked:
	x,y = loc
	visualizer.set_blocked(x,y)
for loc in environment.is_terminal:
	x,y = loc
	reward = environment.rewards.get(loc, 0)
	visualizer.set_value(x,y,reward)

# Draw the initial state
visualizer.draw()

# Create an agent
agent = ValueIterationAgent(environment)

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