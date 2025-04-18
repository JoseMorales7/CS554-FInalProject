import networkx as nx
import pandas as pd
import os

# Load the graph
graph_path = "data/graph_with_communities.graphml"
print(f"Loading graph from {graph_path}...")
G = nx.read_graphml(graph_path)

# Fallback to undirected for modularity computation
if G.is_directed():
    G = G.to_undirected()

# Extract all node IDs
all_nodes = set(G.nodes())

# Extract communities from node attributes
communities = {}
for node, data in G.nodes(data=True):
    comm_id = data.get("community")
    if comm_id is not None:
        communities.setdefault(comm_id, set()).add(node)

# Combine communities into list of sets
community_list = list(communities.values())

# Validate: all nodes must be assigned
assigned_nodes = set().union(*community_list)
unassigned_nodes = all_nodes - assigned_nodes

if unassigned_nodes:
    print(f"Warning: {len(unassigned_nodes)} unassigned nodes found. Assigning them to singleton communities.")
    for node in unassigned_nodes:
        community_list.append({node})

# Compute modularity
print("Computing modularity...")
modularity_score = nx.algorithms.community.quality.modularity(G, community_list)
print(f"Modularity: {modularity_score:.4f}")

# Save to file
os.makedirs("output", exist_ok=True)
with open("output/modularity_score.txt", "w") as f:
    f.write(f"Modularity Score: {modularity_score:.4f}\n")
