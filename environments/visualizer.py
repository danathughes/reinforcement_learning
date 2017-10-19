import pygame
from pygame.locals import *

import numpy as np

class GridworldVisualizer:
   """
   """

   def __init__(self, width, height, cell_width = 10, cell_height = 10, max_value = 100):
   	"""
   	"""

   	pygame.init()

   	self.width = width
   	self.height = height

   	self.cell_width = cell_width
   	self.cell_height = cell_height

   	self.values = np.zeros((self.width, self.height))
   	self.blocks = set()

   	self.max_value = max_value

   	self.agent = (0,0)

   	self.screen = pygame.display.set_mode((cell_width*width+1, cell_height*height+1))


   def set_blocked(self, x, y):
   	"""
   	"""

   	self.blocks.add((x,y))


   def set_value(self, x, y, value):
   	"""
   	"""

   	self.values[x,y] = value


   def set_agent_loc(self, x, y):
   	"""
   	"""

   	self.agent = (x,y)


   def draw(self):
   	"""
   	"""

   	for x in range(self.width):
   		for y in range(self.height):

   			r = pygame.Rect(self.cell_width*x, self.cell_height*y, self.cell_width, self.cell_height)

   			lvl = min(abs(int(255*self.values[x,y] / self.max_value)), 255)

   			if self.values[x,y] > 0:
   				c = pygame.Color(0, lvl, 0, 0)
   			elif self.values[x,y] < 0:
   				c = pygame.Color(lvl, 0, 0, 0)
   			else:
   				c = pygame.Color(0,0,0,0)
   			pygame.draw.rect(self.screen, c, r)

   	# Draw blocked areas as grey
   	for loc in self.blocks:
   		x,y = loc
   		r = pygame.Rect(self.cell_width*x, self.cell_height*y, self.cell_width, self.cell_height)
   		pygame.draw.rect(self.screen, Color(128,128,128,0), r)

   	# Draw a circle where the agent is
   	x,y = self.agent
   	cx = x*self.cell_width + self.cell_width / 2
   	cy = y*self.cell_width + self.cell_width / 2
   	rad = min(self.cell_width, self.cell_height) / 2
   	circ = pygame.draw.circle(self.screen, Color(255,255,0), (cx,cy), rad)

   	pygame.display.flip()


