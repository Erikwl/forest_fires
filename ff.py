import numpy as np
import random
from pyics import Model

import matplotlib.pyplot as plt
from pixelize_image import image

"""
States:
0: sand
1: gras/bush
2: trees
3: burning gras/bush
4: burning trees
5: burned down
"""

BURNING_RED = np.array([217, 63, 63], dtype=np.uint8)
TREE_GREEN = np.array([63, 92, 59], dtype=np.uint8)
GRAS_GREEN = np.array([135, 166, 131], dtype=np.uint8)

SAND_WHITE = np.array([213, 230, 204], dtype=np.uint8)
BURNED_BLACK = np.array([0, 0, 0], dtype=np.uint8)

state_colors = [SAND_WHITE, GRAS_GREEN, TREE_GREEN, BURNING_RED, BURNED_BLACK]
burn_time = [0, 1, 2] # Burn time of sand, gras and trees

class CASim(Model):
    def __init__(self):
        Model.__init__(self)

        self.t = 0
        self.image_name = 'place.png'
        self.size = 200
        self.run_time = 200

        self.height = self.size
        self.width = self.size

        sand = (23, 31, 83, 89)
        gras = (130, 160, 130, 160)
        woods = (75, 120, 110, 155)

        self.areas = np.array([sand, gras, woods])
        self.areas = np.array(list(list(map(int, area * self.size / 200)) for area in self.areas))

    def setup_initial_field(self):
        """Returns an array of length `width' with the initial state for each of
        the cells in the first row. Values should be between 0 and k."""

        self.image = image(self.areas, self.image_name)

        self.image.resize_image(self.size)
        self.image.classify_pixels()

        return self.image.classified_pixels

    def reset(self):
        """Initializes the configuration of the cells and converts the entered
        rule number to a rule set."""
        states = self.setup_initial_field()

        self.config = [[{'state' : int(states[row][col]), 'burn_time' : 0}
                        for col in range(self.width)]
                       for row in range(self.height)]

        self.config[0][0]['state'] = 3
        self.burning = set([(0, 0)])

        self.color_image = np.array([[state_colors[self.config[row][col]['state']]
                                      for col in range(self.width)]
                                     for row in range(self.height)])

        self.t = 0


    def draw(self):
        """Draws the current state of the grid."""

        plt.cla()
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.imshow(self.color_image)
        plt.axis('image')
        plt.title('t = %d' % self.t)

    def step(self):
        """Performs a single step of the simulation by advancing time (and thus
        row) and applying the rule to determine the state of the cells."""
        self.t += 1
        if self.t >= self.run_time:
            return True

        for row, col in list(self.burning):
            for nbr_row in range(max(row - 1, 0), min(row + 2, self.height - 1)):
                for nbr_col in range(max(col - 1, 0), min(col + 2, self.width)):
                    nbr = self.config[nbr_row][nbr_col]
                    if random.random() < 0.5 and nbr['state'] in [1, 2]:
                        nbr['burn_time'] = burn_time[nbr['state']]
                        nbr['state'] += 2
                        self.color_image[nbr_row][nbr_col] = 3
                        self.burning.add((nbr_row, nbr_col))
            item = self.config[row][col]
            item['burn_time'] -= 1
            if item['burn_time'] == 0:
                self.burning.remove((row, col))
                item['state'] = 5
                self.color_image[row][col] = state_colors[4]


if __name__ == '__main__':
    sim = CASim()
    from pyics import GUI
    cx = GUI(sim)
    cx.start()
