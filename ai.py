import numpy as np
import sapai as sp
import scipy

TEAM_SLOTS = 5
PETS_AVAIL = 10
STATUSES = 1
ITEMS_AVAIL = 2
ROUNDS = 2


def relu(x):
    return max(0, x)


def softmax(x):
    return scipy.special.softmax(x)


def generate_wt(x, y):
    l = np.random.rand(x*y)
    l.reshape([x, y])
    return l


class AI:
    def __init__(self, w1=None, w2=None, w3=None):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        if not self.w1:
            self.w1 = generate_wt(91, 30)
        if not self.w2:
            self.w2 = generate_wt(30, 30)
        if not self.w3:
            self.w3 = generate_wt(30, 94)

    def forward(self, input_data):
        m1 = np.dot(input_data, self.w1)
        m2 = np.dot(m1, self.w2)
        o = np.dot(m2, self.w3)
        o = scipy.special.softmax(o)
        return o

"""
State data format:
team_slots(pets_avail+2+statuses)-1
    Team pet information
3(pets_avail+items_avail) 65
    Shop information
Round # 101
Gold # 102

The input used is a numpy array of dimensions
[1,101] which represents the current state of the state
of the game. 
The description goes by the format:

[indices] *the purpose of this set of indices*
index: *what the value at this specific index represents*
... (dots show that the rest of the values of this section
continue with this pattern)

[0-29] actions on shop pets
0: whether tiers.pet_tier_lookup_std[1][0] can be bought
1: if the above is 1, whether 


mask data format:
pets_avail(buy+freeze+unfreeze)
0
    actions on shop pets
items_avail(team_slots+freeze+unfreeze)
 30
    actions on shop items
moves 
44
    ways team pets can be moved
combine 
64
    ways where two team pets can be combined
team_slots(sell, upgrade) 
84
    actions on team pets
turn_end
94
    action to end the turn
"""
def filter_mask(state):
    mask = np.zeros([95, 1])
    mask[94, 0] = 1
    # Can only buy pets if empty space exists and gold >= 3
    empty_slots = []
    for i in range(TEAM_SLOTS):
        if state[0, 11+13*i] == 0:  #  checks if spot is empty
            empty_slots.append(i)
        else:
            # enables sell action for pet
            mask[84 + 2*i] = 1

            for slot in range(TEAM_SLOTS):
                for pet in range(PETS_AVAIL):
                    # checks if same pet is in shop
                    if state[0, pet + 13 * slot] and state[0, 65 + 2*pet]:
                        # enables upgrade action for pet
                        mask[85 + 2*slot] = 1

            # checks which items are available to buy for the pet
            for item in range(ITEMS_AVAIL):
                if state[0, 95 + 3*item]:
                    if state[96 + 3*item]:  # checks if the items are frozen or not
                        mask[35 + 6*item] = 1
                    else:
                        mask[36 + 6*item] = 1

                if state[0, 101] >= state[0, 97 + 3*item]:
                    mask[30 + i + 6*item] = 1

    for i in range(PETS_AVAIL):  # checks which pets are in the shop
        if state[0, 65 + 3*i]:
            if state[0, 67 + 3*i] <= state[0, 102] and len(empty_slots) > 0:  # checks if enough money to buy
                mask[3*i, 0] = 1
            if state[0, 66 + 3*i] == 0:  # checks if they are frozen
                mask[3*i+1, 0] = 1
            else:  # assumes unfrozen otherwise
                mask[3*i+2, 0] = 1
        else:
            mask[3*i, 0] = 0

    move = 0
    for slot in range(TEAM_SLOTS):
        for slot2 in range(slot+1, TEAM_SLOTS):
            for pet in range(PETS_AVAIL):
                # checks if two slots are different
                if state[0, pet + 13 * slot] != state[0, pet + 13 * slot2]:
                    mask[44 + move] = 1
                    break
                # checks if slots are both nonempty
                elif state[0, pet + 13 * slot]:
                    mask[64 + move] = 1
                    break
            move += 1

    return mask


