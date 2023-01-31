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
# print(team[0])

# print(p.state['team']['team'][0]['pet'])
# print(p.state['shop']['slots'][0])


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

    for shop_slot in shop['slots']:
        shop_item = shop_slot['obj']
        if shop_slot['slot_type'] == 'pet':
            res[curr + 3*tiers.pet_tier_lookup_std[1].index(
                shop_item['name'])] = 1
            # if any frozen pets exist, then the pet will be thought as frozen
            res[curr + 1 + 3*tiers.pet_tier_lookup_std[1].index(
                shop_item['name'])] = 1 if shop_slot['frozen'] else 0
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
