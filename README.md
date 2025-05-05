# CS554-FinalProject (Graph-Based and Semantic Paper Recommendation)

This project explores two complementary approaches for scientific article retrieval: a **Text-based PageRank framework** and a **Graph Neural Network (GNN)-based embedding alignment method**. Together, they demonstrate how semantic similarity and graph structure can be combined for effective and flexible paper recommendation.

---

## 🔍 Approaches

### 1. Text-Based PageRank

This method recommends semantically similar papers using textual metadata such as titles, abstracts, and keywords. The process includes:

- Generating **semantic embeddings** for each paper using a pretrained Sentence Transformer (`all-MiniLM-L6-v2`).
- Clustering papers using **K-means** based on embedding similarity.
- Creating a **semantic similarity graph** within each cluster by connecting papers exceeding a cosine similarity threshold.
- Running **PageRank** on this semantic graph to rank papers based on centrality.
- Separately building a **citation graph** from the dataset’s Citations field and applying PageRank to compute a citation-based score.
- **Final Score = 0.9 × Semantic PR + 0.1 × Citation PR**

This hybrid score balances content similarity with citation importance.

### 2. GNN-Based Embedding Alignment

To incorporate structural information from the citation network and support flexible user queries, we apply a deep learning approach:

- Construct a **citation graph**, where each node is a BioBERT-based Sentence Transformer embedding.
- Train a **Graph Convolutional Network (GCN)** to predict semantic clusters, allowing node representations to incorporate neighborhood structure.
- Freeze the GCN and train a **Multi-Layer Perceptron (MLP)** to **project text-only embeddings into the GCN’s feature space**, using cosine similarity as a training objective.

This enables **retrieval using free-text user queries**, not just articles in the training graph.

---

## 🧠 Key Components

- Citation Graph Construction
- SentenceTransformer Embeddings (BioBERT + MiniLM)
- GCN + Classifier Training
- MLP Projector for Embedding Alignment
- PageRank-Based Hybrid Ranking
