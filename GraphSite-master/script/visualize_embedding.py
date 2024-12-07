import torch
import numpy as np
import os
import matplotlib.pyplot as plt

def load_latent_representations(file_prefix="latent_representations_", file_extension=".pt"):
    """
    Load latent representations saved in PyTorch format.

    Args:
        file_prefix (str): The prefix of the file names (e.g., "latent_representations_").
        file_extension (str): The file extension (e.g., ".pt").
    
    Returns:
        list: A list of NumPy arrays representing the latent embeddings.
    """
    embeddings = []
    file_index = 0
    
    while True:
        file_path = f"{file_prefix}{file_index}{file_extension}"
        if not os.path.exists(file_path):
            break
        # Load the embedding and convert to NumPy
        tensor = torch.load(file_path)
        embeddings.append(tensor.numpy())
        file_index += 1
    
    return embeddings

def visualize_latent_representations(embeddings, max_display=5):
    """
    Visualize latent representations by printing and plotting.

    Args:
        embeddings (list): List of NumPy arrays containing latent representations.
        max_display (int): Maximum number of embeddings to display and plot.
    """
    print(f"Loaded {len(embeddings)} latent representations.")

    for idx, embedding in enumerate(embeddings[:max_display]):
        print(f"\nEmbedding {idx}: Shape {embedding.shape}\n{embedding}")
        
        # Plot the embedding (flattened)
        plt.figure(figsize=(10, 2))
        plt.title(f"Visualization of Latent Representation {idx}")
        plt.plot(embedding.flatten(), marker="o", linestyle="None", markersize=2)
        plt.xlabel("Dimension")
        plt.ylabel("Value")
        plt.show()

if __name__ == "__main__":
    # Load embeddings
    embeddings = load_latent_representations()

    # Visualize embeddings
    visualize_latent_representations(embeddings)
