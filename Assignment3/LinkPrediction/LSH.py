import torch
import torch.nn as nn
import torch.nn.functional as F

class RandomLSH:
    def __init__(self, input_dim, num_hashes):
        self.num_hashes = num_hashes
        self.hyperplanes = torch.randn(num_hashes, input_dim)

    def hash(self, embeddings):
        self.hyperplanes = self.hyperplanes.to(embeddings.device)
        projections = embeddings @ self.hyperplanes.T  # Shape: (num_nodes, num_hashes)
        return (projections > 0).int()  # Convert to binary hash codes (0 or 1)

    def group_by_hash(self, hash_codes):
        buckets = {}
        for i, hash_code in enumerate(hash_codes):
            key = tuple(hash_code.tolist())  # Convert tensor to a hashable tuple
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(i)
        return buckets

    def infer(self, query_embedding, buckets, embeddings, top_k=10):
        query_hash = self.hash(query_embedding.unsqueeze(0))  # Shape: (1, num_hashes)
        query_bucket = tuple(query_hash[0].tolist())

        # Retrieve nodes in the same bucket
        candidate_indices = buckets.get(query_bucket, [])

        # Compute cosine similarity with candidates
        candidate_embeddings = embeddings[candidate_indices]
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0), candidate_embeddings
        )

        # Sort candidates by similarity and retrieve top-K
        top_k_indices = torch.argsort(similarities, descending=True)[:top_k]
        return [candidate_indices[i] for i in top_k_indices]


class TrainableLSH(nn.Module):
    def __init__(self, input_dim, num_hashes):
        """
        Trainable LSH module.

        Args:
        - input_dim (int): Dimensionality of input embeddings.
        - num_hashes (int): Number of hash functions (projections).
        """
        super(TrainableLSH, self).__init__()
        self.num_hashes = num_hashes
        self.hash_proj = nn.Linear(input_dim, num_hashes)  # Trainable projections
        self.sigmoid = nn.Sigmoid()  # For binary hashing

    def forward(self, embeddings):
        # Compute projections
        projections = self.hash_proj(embeddings)  # Shape: (batch_size, num_hashes)

        # Convert to binary hash codes
        hash_codes = (self.sigmoid(projections) > 0.5).float()  # Threshold at 0.5
        return hash_codes

    def bucketize(self, embeddings):
        # Compute binary hash codes
        hash_codes = self.forward(embeddings)  # Shape: (batch_size, num_hashes)

        # Convert binary hash codes to bucket indices
        bucket_indices = hash_codes.matmul(2**torch.arange(self.num_hashes, device=embeddings.device).float())

        # Group node indices by bucket
        buckets = {}
        for node_idx, bucket_idx in enumerate(bucket_indices.tolist()):
            if bucket_idx not in buckets:
                buckets[bucket_idx] = []
            buckets[bucket_idx].append(node_idx)
        return buckets

    def infer(self, query_embedding, buckets, embeddings, top_k=10):
        query_hash_code = self.forward(query_embedding)  # Shape: (batch_size, num_hashes)
        query_bucket = query_hash_code.matmul(2**torch.arange(self.num_hashes, device=embeddings.device).float())

        candidate_indices = buckets.get(query_bucket, [])

        candidate_embeddings = embeddings[candidate_indices]
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0), candidate_embeddings
        )

        top_k_indices = torch.argsort(similarities, descending=True)[:top_k]
        return [candidate_indices[i] for i in top_k_indices]

    def train(self, embeddings, margin=1.0, epochs=10, batch_size=32, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        num_samples = embeddings.size(0)

        for epoch in range(epochs):
            total_loss = 0.0
            permutation = torch.randperm(num_samples)  # Shuffle the data

            for i in range(0, num_samples, batch_size):
                batch_indices = permutation[i:i + batch_size]
                batch_embeddings = embeddings[batch_indices]

                # Forward pass to compute hash codes
                batch_hash_codes = self.forward(batch_embeddings)

                # Pairwise similarities
                similarities = torch.nn.functional.cosine_similarity(
                    batch_embeddings.unsqueeze(1), batch_embeddings.unsqueeze(0), dim=-1
                )

                # Positive pairs (high similarity, should be in the same bucket)
                pos_mask = similarities > 0.8  # Define a threshold for positive pairs
                pos_pairs = batch_hash_codes[pos_mask].view(-1, self.num_hashes)

                # Negative pairs (low similarity, should not be in the same bucket)
                neg_mask = similarities < 0.2  # Define a threshold for negative pairs
                neg_pairs = batch_hash_codes[neg_mask].view(-1, self.num_hashes)

                # Loss: Contrastive loss to maximize separation
                if pos_pairs.size(0) > 0 and neg_pairs.size(0) > 0:
                    pos_loss = (1 - pos_pairs).sum(dim=-1).mean()  # Encourage similar hashes
                    neg_loss = torch.relu(margin + neg_pairs.sum(dim=-1)).mean()  # Push apart dissimilar hashes
                    loss = pos_loss + neg_loss
                else:
                    loss = torch.tensor(0.0, device=embeddings.device)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

# Example Usage
if __name__ == "__main__":
    import os
    input_dim = 32  # Dimensionality of node embeddings
    num_hashes = 8  # Number of hash functions

    lsh = TrainableLSH(input_dim, num_hashes)

    torch.save(lsh.state_dict(), args.trained_lsh)

    # Generate random embeddings
    embeddings = torch.randn(32, input_dim)  # Batch of 32 embeddings

    # Compute hash codes and bucket indices
    hash_codes = lsh(embeddings)
    buckets = lsh.bucketize(embeddings)

    print("Hash Codes:\n", hash_codes)
    print("Bucket Indices:\n", buckets)
