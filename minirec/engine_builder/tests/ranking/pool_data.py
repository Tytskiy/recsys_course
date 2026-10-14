"""Constants shared by the ranking pool-builder tests (read from the dm_action fixture)."""

USER_ID = 4298113594485736686
ITEM_ID = 3251178016424488087

# Timestamps in dm_action for USER_ID + ITEM_ID (epoch seconds):
#   1696774707  AT_CartUpdate
#   1696774728  AT_View        (+21 s)
#   1696775107  AT_Purchase    (+400 s)
CART_UPDATE_TS = 1696774707

POSITIVE_ACTIONS = ["AT_CartUpdate", "AT_Purchase"]
NEGATIVE_ACTIONS = ["AT_View"]
