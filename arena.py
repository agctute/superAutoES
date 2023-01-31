from ai import AI
from convert import conv_to_arr
import numpy as np

"""
Object representing a group of players playing SAP
@param pop number of players in this game
@param rounds maximum number of rounds played per tournament
@param tournaments maximum number of tournaments played in one run
@param players list of AI objects for this game
"""
class Arena:
    def __init__(self, pop, rounds, tournaments, robots=None):
        self.pop = pop
        self.rounds = rounds
        self.tournaments = tournaments
        self.robots = robots
        if not robots:
            self.robots = np.empty(pop, AI)
            for i in range(pop):
                self.robots[i] = AI()
        elif len(robots) < pop:  # filling in more if robots not enough
            self.robots = np.empty(pop, AI)
            for i in range(len(robots)):
                self.robots[i] = robots[i]
            for i in range(len(robots), pop):
                self.robots[i] = AI()

    def run(self):
        for i in range(10):



