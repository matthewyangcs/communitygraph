import time
from collections import Counter
from functools import cache
from typing import Optional, Union

import community as community_louvain  # type: ignore
import networkx as nx  # type: ignore
import pandas as pd  # type: ignore
from networkx.algorithms import bipartite  # type: ignore


# TODO: Convert debug outputs to logger and allow levels of debug outputs
class BipartiteCommunity:
    """Represents a weighted bipartite community.

    Takes in a dataframe with each row representing an edge between partite sets. 
    Specifically, we use user_key and item_key to denote each partite set.

    The community does not change after it is initialized.

    Note: You cannot specify a min_user_degree AND a min_item_degree at
    the same time. This results in a circular dependency.

    Parameters:
        df: DataFrame representing an edge list
        user_key: Column name containing the user partite set
        item_key: Column name containing the item partite set
        min_user_degree: Do not consider users with lower degree than this.
        min_item_degree: Do not consider items with lower degree than this
    """

    def __init__(
        self,
        df: pd.DataFrame,
        user_key: str,
        item_key: str,
        min_user_degree: Optional[int] = None,
        min_item_degree: Optional[int] = 10,
    ):
        # 1. Init variables
        print("Initializing...")
        self._df = df.copy()
        self._G = nx.Graph()
        self.user_key = user_key
        self.item_key = item_key

        # 2. Apply min_degree filters on dataframe
        print("Filtering dataframe...")
        if min_user_degree and min_item_degree:
            raise Exception("See docstring: Illegal settings")
        elif min_user_degree:
            counts = self._df[user_key].value_counts()
            self._df = self._df.loc[
                self._df[user_key].isin(counts[counts >= min_user_degree].index), :
            ]
        elif min_item_degree:
            counts = self._df[item_key].value_counts()
            self._df = self._df.loc[
                self._df[item_key].isin(counts[counts >= min_item_degree].index), :
            ]

        # 3. Create graph from the filtered nodes
        print("Adding nodes...")
        self.user_degree: dict[Union[int, str], int] = Counter(self._df[user_key])
        self.item_degree: dict[Union[int, str], int] = Counter(self._df[item_key])
        user_nodes = [
            (user, {"origin": "user", "degree": degree})
            for user, degree in self.user_degree.items()
        ]
        item_nodes = [
            (item, {"origin": "item", "degree": degree})
            for item, degree in self.item_degree.items()
        ]
        self._G.add_nodes_from(user_nodes + item_nodes)

        # 4. Add weighted edges based on filtered nodes
        print("Adding edges...")
        edge_weights = Counter(list(zip(self._df[user_key], self._df[item_key])))
        edge_list = [
            (user, item, weight) for (user, item), weight in edge_weights.items()
        ]
        self._G.add_weighted_edges_from(edge_list)
        print("Completed.\n")

    @cache
    def project_onto_items(self) -> nx.Graph:
        print("Starting weighted projection...")
        start = time.time()
        projected = bipartite.weighted_projected_graph(
            self._G, set(self._df[self.item_key])
        )
        print(f"Finished weighted projection in {time.time() - start}\n")
        return projected

    @cache
    def project_onto_users(self) -> nx.Graph:
        print("Starting weighted projection...")
        start = time.time()
        projected = bipartite.weighted_projected_graph(
            self._G, set(self._df[self.user_key])
        )
        print(f"Finished weighted projection in {time.time() - start}\n")
        return projected

    @cache
    def partition_items(self, resolution=1.0) -> dict[Union[int, str], int]:
        print(f"Starting partition of items with resolution {resolution}...")
        start = time.time()
        projected = self.project_onto_items()
        partition = community_louvain.best_partition(
            projected, weight="weight", resolution=resolution
        )
        print(f"Finished partition in {time.time() - start}")
        return partition

    @cache
    def partition_users(self, resolution=1.0) -> dict[Union[int, str], int]:
        print(f"Starting partition of users with resolution {resolution}...")
        start = time.time()
        projected = self.project_onto_users()
        partition = community_louvain.best_partition(
            projected, weight="weight", resolution=resolution
        )
        print(f"Finished partition in {time.time() - start}\n")
        return partition

    # TODO: Make this describe function work with any graph
    def describe_bipartite(self):
        print(f"Total # of edges (interactions): {len(self._df)}\n")

        print(f"# of unique {self.user_key}: {len(self.user_degree)}")
        print(f"# of unique {self.item_key}: {len(self.item_degree)}")
        assert (
            len(self.user_degree) + len(self.item_degree) == self._G.number_of_nodes()
        )
        print(f"# of unique edges: {self._G.number_of_edges()}\n")

        print(
            f"Average {self.user_key} weighted degree: {len(self._df) / len(self.user_degree)}"
        )
        print(
            f"Average {self.item_key} weighted degree: {len(self._df) / len(self.item_degree)}"
        )
        assert sum(self.user_degree.values()) == sum(self.item_degree.values())
        print(
            f"Average edge weight: {sum(self.user_degree.values()) / self._G.number_of_edges()}\n"
        )

    def get_bipartite(self) -> nx.Graph:
        """Returns the underlying networkx graph"""
        return self._G

    def get_df(self) -> nx.Graph:
        """Returns the underlying dataframe"""
        return self._df
