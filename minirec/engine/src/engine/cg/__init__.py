from engine.cg.history import History
from engine.cg.item_to_item_knn import ItemToItemKNN
from engine.cg.item_to_item_lists import ItemToItemLists
from engine.cg.mixing import CANDGEN_MIXERS, MixigenCandidateMixer, SchemaCandidateMixer
from engine.cg.slim import SLIMToHistory
from engine.cg.top_pop import TopPop
from engine.cg.user_to_item_knn import UserToItemKNN
from engine.interfaces import CandidateGenerator

CANDGENS: dict[str, type[CandidateGenerator]] = {
    "TopPop": TopPop,
    "History": History,
    "ItemToItemLists": ItemToItemLists,
    "SLIMToHistory": SLIMToHistory,
    "UserToItemKNN": UserToItemKNN,
    "ItemToItemKNN": ItemToItemKNN,
}

__all__ = [
    "SchemaCandidateMixer",
    "MixigenCandidateMixer",
    "History",
    "ItemToItemLists",
    "TopPop",
    "SLIMToHistory",
    "UserToItemKNN",
    "ItemToItemKNN",
    "CANDGENS",
    "CANDGEN_MIXERS",
]
