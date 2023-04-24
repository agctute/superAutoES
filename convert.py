import numpy as np
from sapai import Player
from sapai import data
from sapai import tiers

TEAM_SLOTS = 5
PETS_AVAIL = len(tiers.pet_tier_lookup_std[1])
STATUSES = 1
ITEMS_AVAIL = len(tiers.food_tier_lookup[1])
ROUNDS = 2

"""
State data format:
team_slots(pets_avail+2+statuses)-1
    Team pet information
3(pets_avail+items_avail) 65
    Shop information
Round # 101
Gold # 102
"""
p = Player()
# TEAM_SLOTS: total number of slots in the team (usually 5)
# PETS_AVAIL: number of pets available in the current round
# STATUSES: number of statuses that a pet can potentially have in the current round
# ITEMS_AVAIL: number of items available in the current round
# ROUND: current round
# GOLD: current gold

# This function converts the player's state into a numpy array, which only considers the first 3 rounds of the game
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
def conv_to_arr(player):
    res = np.zeros(103)
    res[95:100:3] = 3
    curr = 0
    team = player.state['team']['team']
    shop = player.state['shop']
    # Team information
    for team_slot in team:
        if team_slot['pet']['name'] == 'pet-none':
            curr += (PETS_AVAIL + 2 + STATUSES)
        else:
            res[curr + tiers.pet_tier_lookup_std[1].index(
                team_slot['pet']['name'])] = 1
            curr += PETS_AVAIL
            res[curr] = team_slot['pet']['attack']
            curr += 1
            res[curr] = team_slot['pet']['health']
            curr += 1
            if team_slot['pet']['status'] != 'none':
                res[curr] = 1
            curr += 1

    # Shop information  
    for shop_slot in shop['slots']:
        shop_item = shop_slot['obj']
        if shop_slot['slot_type'] == 'pet':
            res[curr + 3*tiers.pet_tier_lookup_std[1].index(
                shop_item['name'])] = 1
            # if any frozen pets exist, then the pet will be thought as frozen
            res[curr + 1 + 3*tiers.pet_tier_lookup_std[1].index(
                shop_item['name'])] = 1 if shop_slot['frozen'] else 0
            # cost of the pet (only 3 now since it doesn't change in the early rounds)
            res[curr + 2 + 3 * tiers.pet_tier_lookup_std[1].index(
                shop_item['name'])] = 3 

        elif shop_slot['slot_type'] == 'food':
            curr_index = tiers.food_tier_lookup[1].index(shop_item['name'])
            res[curr + PETS_AVAIL + 3*curr_index] = 1
            # same logic from pets applies to food
            res[curr + PETS_AVAIL + 1 + 3*curr_index] = 1 if shop_slot['frozen'] else 0
            # change cost
            res[curr + PETS_AVAIL + 2 + 3*curr_index] = min(shop_slot['cost'],
                                                            res[curr + PETS_AVAIL + 2 + 3*curr_index]
                                                            )
    curr += 3*(PETS_AVAIL + ITEMS_AVAIL)
    res[curr] = player.turn
    curr += 1
    res[curr] = player.gold
    return np.expand_dims(res, axis=0)

# p.buy_pet(1)
# p.buy_pet(1)
# p.buy_pet(0)
# print(conv_to_arr(p))

# elif team_slot['pet']['name'] in
