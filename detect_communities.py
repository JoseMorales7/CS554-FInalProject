import os
import networkx as nx
import community as community_louvain
import csv


def load_graph(path):
    """
    Load a graph from a .graphml file.
    """
    return nx.read_graphml(path)


def save_graph_with_communities(G, path):
    """
    Save the graph to a .graphml file after assigning community IDs.
    """
    nx.write_graphml(G, path)


def export_community_assignments(G, output_csv):
    """
    Write each node and its community ID to a CSV file.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)  # <-- this ensures the folder exists
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "community_id"])
        for node in G.nodes():
            writer.writerow([node, G.nodes[node].get("community", -1)])


def detect_communities(graph_path, output_graph_path, output_csv_path):
    print("Loading the graph...")
    G = load_graph(graph_path)

    print("Running Louvain community detection...")
    partition = community_louvain.best_partition(G)

    print("Assigning community IDs to nodes...")
    for node_id, community_id in partition.items():
        G.nodes[node_id]["community"] = community_id

    print("Saving the updated graph with community information...")
    save_graph_with_communities(G, output_graph_path)

    print("Exporting community assignments to CSV...")
    export_community_assignments(G, output_csv_path)

    print("Done. Community detection complete.")


if __name__ == "__main__":
    detect_communities(
        graph_path="data/test_network.graphml",
        output_graph_path="data/graph_with_communities.graphml",
        output_csv_path="output/community_assignments.csv"
    )
