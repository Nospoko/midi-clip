import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from train import preprocess_dataset
from models.pitch_encoder import PitchEncoder
from models.velocity_time_encoder import VelocityTimeEncoder


def visualize_embeddings(
    pitch_encoder: PitchEncoder, velocity_time_encoder: VelocityTimeEncoder, loader: DataLoader, device: torch.device
):
    batch = next(iter(loader))

    pitch = batch["pitch"].to(device)
    dstart_bin = batch["dstart_bin"].to(device)
    duration_bin = batch["duration_bin"].to(device)
    velocity_bin = batch["velocity_bin"].to(device)

    pitch_embeddings = pitch_encoder(pitch).detach().cpu().numpy()
    velocity_time_embeddings = velocity_time_encoder(velocity_bin, dstart_bin, duration_bin).detach().cpu().numpy()

    kmeans = KMeans(n_clusters=10, n_init="auto")
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

    train_dataloader, val_dataloader, test_dataloader = preprocess_dataset(
        "JasiekKaczmarczyk/giant-midi-quantized", batch_size=1024, num_workers=8
    )

    # visualize embeddings on batch from train set
    visualize_embeddings(pitch_encoder, velocity_time_encoder, train_dataloader, device)

    # visualize embeddings on batch from val set
    visualize_embeddings(pitch_encoder, velocity_time_encoder, val_dataloader, device)


if __name__ == "__main__":
    main()
