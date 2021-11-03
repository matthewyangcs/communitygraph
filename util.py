import contextlib
import os
import time
from collections import Counter

# import community as community_louvain  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore
import pandas as pd  # type: ignore
from community.community_louvain import modularity  # type: ignore

from communitygraph.bipartite import BipartiteCommunity  # type: ignore


def label_df_partition(
    df: pd.DataFrame, col_name: str, partition: dict[str, int], inplace=False
) -> pd.DataFrame:
    """Labels the dataframe using the partition"""
    if not inplace:
        df = df.copy()
    df = df[df[col_name].isin(partition)]
    df["community"] = df[col_name].apply(lambda x: partition[x])
    return df


def plot_partition_distribution(partition: dict[str, int]) -> plt.Figure:
    """Plots a histogram from dict"""
    counts = Counter(partition.values())
    fig = plt.figure()

    # Creating the barplot
    plt.bar(counts.keys(), counts.values())
    plt.xlabel("Community ID")
    plt.ylabel("Size")
    plt.title("Partition Distribution")
    return fig


# TODO: When BipartiteCommunity is switched to logger, we no longer need context manager
def optimize_modularity(
    df: pd.DataFrame,
    user_key: str,
    item_key: str,
    min_item_degree: list[int],
    resolution: list[float] = [1.0],
    debug=True,
) -> dict:
    """2d grid search for best min_degree and resolution"""
    df = df.copy()
    data = {}

    print("Starting search over: ")
    print(f" - min_item_degree: {min_item_degree}")
    print(f" - resolution: {resolution}\n")

    iter = 1
    for min_deg in min_item_degree:
        # Don't show debug outputs for creating the community + projection
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            bc = BipartiteCommunity(df, user_key, item_key, min_item_degree=min_deg)
            projected = bc.project_onto_items()

        for res in resolution:
            start = time.time()

            # Don't show debut outputs for partitioning
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                partition = bc.partition_items(resolution=res)
                curr_mod = modularity(partition, projected)

            data[(min_deg, res)] = curr_mod

            if debug:
                print(
                    f"Iteration {iter}/{len(min_item_degree) * len(resolution)}:",
                    f"min_deg {min_deg}, resolution {res}",
                )
                print(f"Modularity: {curr_mod}")
                community_sizes = Counter(partition.values())
                num_nodes = sum(community_sizes.values())
                unique_communities = len(community_sizes)
                counts = sorted(community_sizes.values())
                print(f"Median community size: {counts[len(counts)//2]}")
                print(f"# communities: {unique_communities}")
                print(f"# nodes: {num_nodes}")
                print(f"Time taken: {time.time() - start}\n")
            iter += 1

    return data
