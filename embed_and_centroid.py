import os
import torch
import networkx as nx
import pandas as pd
import numpy as np
import json
from collections import defaultdict
from torch_geometric.utils import from_networkx
from torch_geometric.nn.models import Node2Vec

def run_node2vec():
    print("Loading graph...")
    G = nx.read_graphml("data/graph_with_communities.graphml")
    G = G.to_undirected()

    print("Converting graph to PyTorch Geometric format...")
    data = from_networkx(G)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Node2Vec(
        data.edge_index, embedding_dim=128, walk_length=20,
        context_size=10, walks_per_node=10, num_negative_samples=1,
        sparse=True
    ).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)

    print("Training Node2Vec model...")
    model.train()
    for epoch in range(1, 6):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    model.eval()
    embeddings = model.embedding.weight.cpu().detach().numpy()
    node_ids = list(G.nodes())
    node_embeddings = {node_ids[i]: embeddings[i] for i in range(len(node_ids))}

    df = pd.DataFrame.from_dict(node_embeddings, orient="index")
    df.index.name = "node_id"
    df.to_csv("output/node_embeddings.csv")
    print("Node embeddings saved")

    print("Computing community centroids...")
    communities = nx.get_node_attributes(G, "community")
    vectors = defaultdict(list)
    for node, comm in communities.items():
        if node in node_embeddings:
            vectors[comm].append(node_embeddings[node])

    centroids = {comm: np.mean(vecs, axis=0).tolist() for comm, vecs in vectors.items()}
    with open("output/community_centroids.json", "w") as f:
        json.dump(centroids, f)
    print("Centroids saved")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    run_node2vec()
