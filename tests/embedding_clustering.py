import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import Subset, DataLoader

from data.dataset import MidiDataset
from models.pitch_encoder import PitchEncoder
from models.velocity_time_encoder import VelocityTimeEncoder


def preprocess_dataset(dataset_name: str, split: str, batch_size: int, num_workers: int, *, query: str = None):
    dataset = load_dataset(dataset_name, split=split)
    ds = MidiDataset(dataset)

    if query:
        idx_query = [i for i, name in enumerate(ds.data["midi_filename"]) if str.lower(query) in str.lower(name)]
        ds = Subset(ds, indices=idx_query)
        batch_size = len(ds) if len(ds) < batch_size else batch_size

    # dataloaders
    dataloader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return dataloader


def visualize_embeddings(
    pitch_encoder: PitchEncoder,
    velocity_time_encoder: VelocityTimeEncoder,
    batch: dict,
    device: torch.device,
):
    pitch = batch["pitch"].to(device)
    dstart_bin = batch["dstart_bin"].to(device)
    duration_bin = batch["duration_bin"].to(device)
    velocity_bin = batch["velocity_bin"].to(device)

    with torch.no_grad():
        pitch_embeddings = pitch_encoder(pitch).detach().cpu().numpy()
        velocity_time_embeddings = velocity_time_encoder(velocity_bin, dstart_bin, duration_bin).detach().cpu().numpy()

    kmeans = KMeans(n_clusters=11, n_init="auto")
    pitch_clusters = kmeans.fit(pitch_embeddings).labels_
    # velocity_time_clusters = kmeans.transform(velocity_time_embeddings).labels_

    # dim reduction
    pca = PCA(n_components=2)
    pitch_pca = pca.fit_transform(pitch_embeddings)
    velocity_time_pca = pca.transform(velocity_time_embeddings)

    # color_palette = np.arange(pitch_pca.shape[0]) / pitch_pca.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].scatter(pitch_pca[:, 0], pitch_pca[:, 1], c=pitch_clusters, cmap="tab20c")
    axes[0].set_title("Pitch embeddings")

    axes[1].scatter(velocity_time_pca[:, 0], velocity_time_pca[:, 1], c=pitch_clusters, cmap="tab20c")
    axes[1].set_title("Velocity and time embeddings")

    plt.show()


def visualize_based_on_query(
    pitch_encoder: PitchEncoder, velocity_time_encoder: VelocityTimeEncoder, loader: DataLoader, query: str, device: torch.device
):
    batch = next(iter(loader))

    pitch = batch["pitch"].to(device)
    dstart_bin = batch["dstart_bin"].to(device)
    duration_bin = batch["duration_bin"].to(device)
    velocity_bin = batch["velocity_bin"].to(device)

    with torch.no_grad():
        pitch_embeddings = pitch_encoder(pitch).detach().cpu().numpy()
        velocity_time_embeddings = velocity_time_encoder(velocity_bin, dstart_bin, duration_bin).detach().cpu().numpy()

    # dim reduction
    pca = PCA(n_components=2)
    pitch_pca = pca.fit_transform(pitch_embeddings)
    velocity_time_pca = pca.transform(velocity_time_embeddings)

    # color_palette = np.arange(pitch_pca.shape[0]) / pitch_pca.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f"Query: {query}")
    axes[0].scatter(pitch_pca[:, 0], pitch_pca[:, 1])
    axes[0].set_title("Pitch embeddings")

    axes[1].scatter(velocity_time_pca[:, 0], velocity_time_pca[:, 1])
    axes[1].set_title("Velocity and time embeddings")

    plt.show()


def review_maestro_composer_embeddings(
    pitch_encoder: PitchEncoder,
    velocity_time_encoder: VelocityTimeEncoder,
):
    device = "cpu"
    pitch_encoder.to(device)
    velocity_time_encoder.to(device)

    pitch_encoder.eval()
    velocity_time_encoder.eval()

    val_dataset = load_dataset("roszcz/maestro-quantized", split="validation")
    val_dataset = MidiDataset(val_dataset)

    queries = ["chopin", "bach", "liszt", "rach"]
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for query in queries:
        idx_query = [idx for idx, name in enumerate(val_dataset.dataset["source"]) if str.lower(query) in str.lower(name)]
        subset = Subset(val_dataset, indices=idx_query)
        batch = subset[:]
        pitch = batch["pitch"].to(device)
        dstart_bin = batch["dstart_bin"].to(device)
        duration_bin = batch["duration_bin"].to(device)
        velocity_bin = batch["velocity_bin"].to(device)

        with torch.no_grad():
            pitch_embeddings = pitch_encoder(pitch).detach().cpu().numpy()
            velocity_time_embeddings = velocity_time_encoder(velocity_bin, dstart_bin, duration_bin).detach().cpu().numpy()

        pitch_embeddings /= np.linalg.norm(pitch_embeddings)
        velocity_time_embeddings /= np.linalg.norm(velocity_time_embeddings)

        pca = PCA(n_components=2)
        pitch_pca = pca.fit_transform(pitch_embeddings)
        velocity_time_pca = pca.transform(velocity_time_embeddings)
        axes[0].scatter(pitch_pca[:, 0], pitch_pca[:, 1], label=query)

        axes[1].scatter(velocity_time_pca[:, 0], velocity_time_pca[:, 1], label=query)

    axes[0].set_title("Pitch embeddings")
    axes[1].set_title("Velocity and time embeddings")
    axes[0].legend()
    axes[1].legend()


def main():
    checkpoint = torch.load("checkpoints/midi-clip-batch-1024-2023-08-24-11-10.ckpt")

    cfg = checkpoint["config"]
    device = cfg.train.device

    # model
    pitch_encoder = PitchEncoder(
        num_embeddings=cfg.models.pitch_encoder.num_embeddings,
        embedding_dim=cfg.models.pitch_encoder.embedding_dim,
        output_embedding_dim=cfg.models.pitch_encoder.output_embedding_dim,
        num_attn_blocks=cfg.models.pitch_encoder.num_attn_blocks,
        num_attn_heads=cfg.models.pitch_encoder.num_attn_heads,
        attn_ffn_expansion=cfg.models.pitch_encoder.attn_ffn_expansion,
        dropout_rate=cfg.models.pitch_encoder.dropout_rate,
    ).to(device)

    # forward diffusion
    velocity_time_encoder = VelocityTimeEncoder(
        num_embeddings=cfg.models.velocity_time_encoder.num_embeddings,
        embedding_dim=cfg.models.velocity_time_encoder.embedding_dim,
        output_embedding_dim=cfg.models.velocity_time_encoder.output_embedding_dim,
        num_attn_blocks=cfg.models.velocity_time_encoder.num_attn_blocks,
        num_attn_heads=cfg.models.velocity_time_encoder.num_attn_heads,
        attn_ffn_expansion=cfg.models.velocity_time_encoder.attn_ffn_expansion,
        dropout_rate=cfg.models.velocity_time_encoder.dropout_rate,
    ).to(device)

    pitch_encoder.load_state_dict(checkpoint["pitch_encoder"])
    velocity_time_encoder.load_state_dict(checkpoint["velocity_time_encoder"])

    pitch_encoder.eval()
    velocity_time_encoder.eval()

    dataset_path = "JasiekKaczmarczyk/giant-midi-quantized"

    # visualize embeddings on batch from train set
    train_dataloader = preprocess_dataset(dataset_path, split="train", batch_size=1024, num_workers=8)
    visualize_embeddings(pitch_encoder, velocity_time_encoder, train_dataloader, device)

    # visualize based on query
    query = "Chopin"

    query_dataloader = preprocess_dataset(dataset_path, split="train", batch_size=1024, num_workers=8, query=query)
    visualize_based_on_query(pitch_encoder, velocity_time_encoder, query_dataloader, query=query, device=device)


if __name__ == "__main__":
    main()
