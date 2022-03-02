import numpy as np
from numpy.random import randint, choice, seed
from pyics import Model

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def decimal_to_base_k(n, k):
    """Converts a given decimal (i.e. base-10 integer) to a list containing the
    base-k equivalant.

    For example, for n=34 and k=3 this function should return [1, 0, 2, 1]."""

    result = np.array([])
    while n:
        digit = n % k
        result = np.append(result, digit)
        n //= k
    return result[::-1]


class CASim(Model):
    def __init__(self):
        Model.__init__(self)

        self.t = 0
        self.config = None
        self.set_seed(0)

        self.make_param('r', 1)
        self.make_param('k', 2)
        self.make_param('width', 50)
        self.make_param('height', 50)
        self.make_param('Lambda', 0.0, setter=self.setter_lambda)
        self.make_param('random', False)

    def set_seed(self, random_seed):
        self.seed = random_seed

    def show_colormap(self):
        temp_data = np.random.rand(10,10) * 2 - 1
        plt.pcolor(temp_data, cmap=self.cmap)
        plt.colorbar()
        plt.show()

    def setter_lambda(self, val):
        """Setter for the lambda parameter, changing the given value to the
        closest possible value of lambda."""
        k_N = int(self.k ** (2 * self.r + 1))
        k_N_lam = max(0, min(k_N, round(k_N * val)))
        return k_N_lam / k_N

    def build_rule_set(self):
        """Sets the rule set for the current rule.
        A rule set is a list with the new state for every old configuration."""

        k_N = int(self.k ** (2 * self.r + 1))

        # Check if the new lambda is smaller than the current one or
        # if there is no rule set yet.
        if not len(self.rule_set) or len(self.rule_set) != k_N \
            or sum(0 != self.rule_set) / len(self.rule_set) > self.Lambda:
            self.rule_set = np.zeros(k_N)
        cur_lam = sum(0 != self.rule_set) / len(self.rule_set)

        # Choose items that are going to be changed.
        zero_items = [i for i in range(k_N) if self.rule_set[i] == 0]
        nr_items_to_chose = int((self.Lambda - cur_lam) * k_N)
        chosen_items = choice(zero_items, size=nr_items_to_chose, replace=False)

        # Change the items.
        for index in chosen_items:
            self.rule_set[index] = choice(range(1, self.k))


    def check_rule(self, inp):
        """Returns the new state based on the input states.

        The input state will be an array of 2r+1 items between 0 and k, the
        neighbourhood which the state of the new cell depends on."""

        # Transform the list into an integer.
        combined_state = int(sum(state * self.k ** i \
                                 for i, state in enumerate(inp[::-1])))
        state_index = - combined_state - 1
        return self.rule_set[state_index]

    def setup_initial_state(self):
        """Returns an array of length `width' with the initial state for each of
        the cells in the first row. Values should be between 0 and k."""

        

        return result

    def reset(self):
        """Initializes the configuration of the cells and converts the entered
        rule number to a rule set."""

        # Only reset the values up to the current time.
        if self.t > 0 and self.t < self.height:
            self.config[:self.t + 1] = np.zeros([self.t + 1, self.width])
        else:
            self.config = np.zeros([self.height, self.width])

        self.t = 0

        self.config[0, :] = self.setup_initial_state()
        self.build_rule_set()

    def draw(self):
        """Draws the current state of the grid."""

        plt.cla()
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.imshow(self.config, interpolation='none', vmin=0, vmax=self.k - 1,
                cmap=self.cmap)
        plt.axis('image')
        plt.title('t = %d' % self.t)

    def step(self):
        """Performs a single step of the simulation by advancing time (and thus
        row) and applying the rule to determine the state of the cells."""
        self.t += 1
        if self.t >= self.height:
            return True

        for patch in range(self.width):
            # We want the items r to the left and to the right of this patch,
            # while wrapping around (e.g. index -1 is the last item on the row).
            # Since slices do not support this, we create an array with the
            # indices we want and use that to index our grid.
            indices = [i % self.width
                    for i in range(patch - self.r, patch + self.r + 1)]
            values = self.config[self.t - 1, indices]
            self.config[self.t, patch] = self.check_rule(values)


if __name__ == '__main__':
    sim = CASim()
    sim.show_colormap()
    from pyics import GUI
    cx = GUI(sim)
    cx.start()
