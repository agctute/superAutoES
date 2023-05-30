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

# TEAM_SLOTS: total number of slots in the team (usually 5)
# PETS_AVAIL: number of pets available in the current round
# STATUSES: number of statuses that a pet can potentially have in the current round
# ITEMS_AVAIL: number of items available in the current round
# ROUND: current round
# GOLD: current gold

# This function applies a filter mask to the end output
# and limits options an ai can pick
# The array is of shape (1, 103)
# The array is of the following format, where capital connected words denote
# variables or features from the Player class:
# the first TEAM_SLOTS(PETS_AVAIL + 2 + STATUSES) - 1 entries the state of pets in the team
# the ith (PET_AVAIL + 2 + STATUSES) entries represent the state of the pet in the ith slot, 
# with 5 total potential pets. if the slot is empty, all entries are 0
# This pet state is represented by the name of the pet, its health and attack, and what status it has
# the name of the pet is one-hot encoded, so are the statuses.
# 
# The next 3(PETS_AVAIL + ITEMS_AVAIL) entries represent the state of pets in the shop.
# Since we are only considering the first three rounds, thats was the 3 is for.
# the ith (PET_AVAIL + 2 + STATUSES) entries represent the state of the pet in the ith slot, 
# for every pet and item that can be in the shop, the "3" declares:
# the number of pets/foods of that name in the shop
# whether one of the pets is frozen
# the cost of the pet
# The last two entries are the round number and the gold number

def filter_mask(state):
    """Creates a filter mask for NN output based on given state

    Args:
        data (ndarray): The input data array of shape (1, 103).

    Returns:
        ndarray: The corresponding filter mask

    Input Data format:
    The input data array follows a specific format, where the capital connected words denote
    variables or features from the Player class.

    - The first TEAM_SLOTS (PETS_AVAIL + 2 + STATUSES) - 1 entries represent the state of pets in the team.
      Each (PETS_AVAIL + 2 + STATUSES) entries represents the state of the pet in the corresponding slot,
      with a total of 5 potential pets.
      If a slot is empty, all entries are 0.
      The pet state includes the name of the pet, its health and attack, and any associated status.
      The pet name is one-hot encoded, as are the statuses.

    - The next 3(PETS_AVAIL + ITEMS_AVAIL) entries represent the state of pets in the shop.
      Since we are considering only the first three rounds, the number 3 indicates the No. of rounds.
      Each entry represents the state of the pet or item in the corresponding slot.
      For every pet and item that can be in the shop, the entry contains the following information:
      - The number of pets/foods of that name in the shop.
      - Whether one of the pets/foods is frozen.
      - The cost of the pet/food.

    - The last two entries are the round number and the gold number.

    Output Data format:
    The mask data format specifies the actions available for different components:
    - PETS_AVAIL (BUY+FREEZE+UNFREEZE): indices 0-29 (10 pets)
      Potential actions on shop pets, where every 3 entries represent the option to buy, freeze, or unfreeze
      the ith pet

    - ITEMS_AVAIL (TEAM_SLOTS+freeze+unfreeze): indices 30-43 (2 items, 5 slots)
      Potential actions on shop pets, where every 7 entries represent the option to buy for the ith pet,
      or freeze/unfreeze the ith item

    - MOVES: indices 44-58 (5 + 4 + 3 + 2 + 1)
      Ways team pets can be swapped.
      i.e. swap 1 and 2, 1 and 3, ..., 4 and 5

    - COMBINE: indices 59-73
      Ways where two team pets can be combined.

    - TEAM_SLOTS(sell, upgrade): indices 74-83
      Actions on team pets.

    - turn_end: 84
      Action to end the turn.
    """
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
                    if state[0, 96 + 3*item]:  # checks if the items are frozen or not
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


