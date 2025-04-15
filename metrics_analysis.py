import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

# Load community assignments
df = pd.read_csv("output/community_assignments.csv")

# Compute community size statistics
community_sizes = df["community_id"].value_counts().sort_values(ascending=False)

# Save stats to a text file
with open("output/community_stats.txt", "w") as f:
    f.write(f"Total communities: {community_sizes.shape[0]}\n")
    f.write(f"Largest community size: {community_sizes.iloc[0]}\n")
    f.write(f"Smallest community size: {community_sizes.iloc[-1]}\n")
    f.write(f"Median size: {community_sizes.median()}\n")
    f.write(f"Mean size: {community_sizes.mean():.2f}\n")

print("Community size statistics saved to output/community_stats.txt")

# Save size distribution to CSV
community_sizes.to_csv("output/community_sizes.csv", header=["size"])
print("Community size distribution saved to output/community_sizes.csv")

# Save community summary (top N with IDs and sizes)
top_n = 50
df_summary = community_sizes.head(top_n).reset_index()
df_summary.columns = ["community_id", "size"]
df_summary.index = df_summary.index + 1  # rank starts from 1
df_summary.to_csv("output/community_summary.csv", index_label="rank")
print(f"Top {top_n} community summary saved to output/community_summary.csv")

# Plot top N largest communities
plt.figure(figsize=(12, 6))
community_sizes.head(top_n).plot(kind="bar")
plt.title(f"Top {top_n} Community Sizes")
plt.xlabel("Community ID")
plt.ylabel("Number of Nodes")
plt.tight_layout()
plt.savefig("output/top_communities.png")
plt.show()

# Load community centroids
with open("output/community_centroids.json", "r") as f:
    centroids = json.load(f)

# Analyze centroid statistics
centroid_matrix = np.array(list(centroids.values()))
means = np.mean(centroid_matrix, axis=0)
stds = np.std(centroid_matrix, axis=0)

# Save centroid stats
with open("output/centroid_stats.txt", "w") as f:
    f.write(f"Embedding dimensions: {centroid_matrix.shape[1]}\n")
    f.write("Mean of first 5 dimensions: " + ", ".join(f"{v:.4f}" for v in means[:5]) + "\n")
    f.write("Std of first 5 dimensions: " + ", ".join(f"{v:.4f}" for v in stds[:5]) + "\n")

print("Centroid embedding stats saved to output/centroid_stats.txt")