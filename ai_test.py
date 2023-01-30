import numpy as np
import sapai as sp
from ai import filter_mask
import unittest
from convert import conv_to_arr

# test = np.random.rand(89)
# test = test[np.newaxis]
# for i in range(test.shape[1]):
#     if test[0, i] > 0.5:
#         test[0, i] = 1
#     else:
#         test[0, i] = 0
#
# fido = AI()
#
# print(fido.forward(test))

"""
State data format:
team_slots(pets_avail+2+statuses)-1
    Team pet information
2(pets_avail+items_avail) 65
    Shop information
Round # 90
Gold # 89
"""

# t = np.zeros(89)
# t = np.expand_dims(t, axis=0)
# print(t.shape)


class ConversionTests(unittest.TestCase):
    def test_conversion(self):
        team = sp.Team()
        shop = sp.Shop()
        test_player = sp.Player(shop=shop, team=team)
        test_player.buy_pet(1)
        print(team, shop)
        # print(test_player.state)
        print(conv_to_arr(test_player))


if __name__ == '__main__':
    unittest.main()
