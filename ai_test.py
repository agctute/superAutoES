import numpy as np
import sapai as sp
from ai import filter_mask
import unittest
from convert import conv_to_arr
from ai import filter_mask

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
    def setUp(self) -> None:
        team = sp.Team()
        shop = sp.Shop()
        self.test_player = sp.Player(shop=shop, team=team)
        self.test_player.buy_pet(1)
        print(team, shop)
        self.conv = conv_to_arr(self.test_player)

    def test_conversion(self):
        # print(test_player.state)
        print(conv_to_arr(self.test_player))

    def test_filter(self):
        print(filter_mask(self.conv))


if __name__ == '__main__':
    unittest.main()
