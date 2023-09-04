import os

import torch
import numpy as np
import pretty_midi
import streamlit as st
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
from datasets import Dataset, load_dataset
from torch.utils.data import Subset, DataLoader
from fortepyan.audio import render as render_audio
from huggingface_hub.file_download import hf_hub_download

from data.dataset import MidiDataset
from data.quantizer import MidiQuantizer
from models.pitch_encoder import PitchEncoder
from models.velocity_time_encoder import VelocityTimeEncoder


def preprocess_dataset(dataset: Dataset, batch_size: int, num_workers: int, *, query: str = None):
    ds = MidiDataset(dataset)

    if query:
        idx_query = [i for i, name in enumerate(ds.dataset["source"]) if str.lower(query) in str.lower(name)]
        ds = Subset(ds, indices=idx_query)
        batch_size = len(ds) if len(ds) < batch_size else batch_size

    # dataloaders
    dataloader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return dataloader, ds


def visualize_embeddings(
    pitch_embeddings: np.ndarray,
    velocity_time_embeddings: np.ndarray,
    query_pitch: str,
    query_velocity_time: str,
    pitch_idx: int,
    velocity_time_idx: int,
):
    # dim reduction
    pca = PCA(n_components=2)
    pitch_pca = pca.fit_transform(pitch_embeddings)
    velocity_time_pca = pca.transform(velocity_time_embeddings)

    # color_palette = np.arange(pitch_pca.shape[0]) / pitch_pca.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].scatter(pitch_pca[:, 0], pitch_pca[:, 1], alpha=0.6)
    axes[0].scatter(pitch_pca[pitch_idx, 0], pitch_pca[pitch_idx, 1], c="red")
    axes[0].set_title(f"Pitch embeddings, query: {query_pitch}")

    axes[1].scatter(velocity_time_pca[:, 0], velocity_time_pca[:, 1], alpha=0.6)
    axes[1].scatter(velocity_time_pca[velocity_time_idx, 0], velocity_time_pca[velocity_time_idx, 1], c="red")
    axes[1].set_title(f"Velocity and time embeddings, query: {query_velocity_time}")

    st.pyplot(fig)


def find_closest_embeddings(pitch_embeddings: np.ndarray, velocity_time_embeddings: np.ndarray):
    tree = KDTree(pitch_embeddings)

    pitch_distances, pitch_indices = tree.query(velocity_time_embeddings)

    velocity_time_idx = np.argmin(pitch_distances)
    pitch_idx = pitch_indices[velocity_time_idx]

    return pitch_idx, velocity_time_idx


def embeddings_based_on_query(
    pitch_encoder: PitchEncoder,
    velocity_time_encoder: VelocityTimeEncoder,
    dataset_name: str,
    query_pitch: str,
    query_velocity_time: str,
    device: torch.device,
):
    dataset = load_dataset(dataset_name, split="validation")
    pitch_dataloader, pitch_ds = preprocess_dataset(dataset, batch_size=1024, num_workers=8, query=query_pitch)
    velocity_time_dataloader, velocity_time_ds = preprocess_dataset(
        dataset, batch_size=1024, num_workers=8, query=query_velocity_time
    )

    pitch_embeddings_list = []
    velocity_time_embeddings_list = []

    # pitch encoding
    for batch in pitch_dataloader:
        pitch = batch["pitch"].to(device)

        with torch.no_grad():
            pitch_embeddings_list.append(pitch_encoder(pitch).detach().cpu().numpy())

    # velocity time encoding
    for batch in velocity_time_dataloader:
        dstart_bin = batch["dstart_bin"].to(device)
        duration_bin = batch["duration_bin"].to(device)
        velocity_bin = batch["velocity_bin"].to(device)

        with torch.no_grad():
            velocity_time_embeddings_list.append(
                velocity_time_encoder(velocity_bin, dstart_bin, duration_bin).detach().cpu().numpy()
            )

    pitch_embeddings = np.concatenate(pitch_embeddings_list, axis=0)
    velocity_time_embeddings = np.concatenate(velocity_time_embeddings_list, axis=0)

    # normalization
    pitch_embeddings /= np.linalg.norm(pitch_embeddings, axis=1, keepdims=True)
    velocity_time_embeddings /= np.linalg.norm(velocity_time_embeddings, axis=1, keepdims=True)

    pitch_idx, velocity_time_idx = find_closest_embeddings(pitch_embeddings, velocity_time_embeddings)

    visualize_embeddings(
        pitch_embeddings,
        velocity_time_embeddings,
        query_pitch,
        query_velocity_time,
        pitch_idx,
        velocity_time_idx,
    )

    return pitch_ds[pitch_idx], velocity_time_ds[velocity_time_idx]


def to_midi(pitch_record: dict, velocity_time_record: dict, track_name: str = "piano"):
    quantizer = MidiQuantizer(7, 7, 7)

    pitch = pitch_record["pitch"].numpy()
    dstart_bin = pitch_record["dstart_bin"].numpy()
    duration_bin = velocity_time_record["duration_bin"].numpy()
    velocity_bin = velocity_time_record["velocity_bin"].numpy()

    track = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0, name=track_name)

    previous_start = 0.0

    for p, v, ds, d in zip(pitch, velocity_bin, dstart_bin, duration_bin):
        velocity = quantizer.bin_to_velocity[v]
        dstart = quantizer.bin_to_dstart[ds]
        duration = quantizer.bin_to_duration[d]

        start = previous_start + dstart
        end = start + duration
        previous_start = start

        note = pretty_midi.Note(
            velocity=int(velocity),
            pitch=int(p),
            start=start,
            end=end,
        )

        piano.notes.append(note)

    track.instruments.append(piano)

    return track


def render_midi_to_mp3(pitch_record: dict, velocity_time_record: dict) -> dict:
    filename = f"{pitch_record['filename']} {velocity_time_record['filename']}"
    midi_filename = os.path.basename(filename)
    mp3_path = midi_filename.replace(".midi", ".mp3")
    mp3_path = midi_filename.replace(".mid", ".mp3")
    mp3_path = os.path.join("tmp", mp3_path)

    if not os.path.exists(mp3_path):
        track = to_midi(pitch_record, velocity_time_record)
        render_audio.midi_to_mp3(track, mp3_path)

    return mp3_path


def main():
    checkpoint = torch.load(hf_hub_download("roszcz/midi-clip", filename="midi-clip-batch-256-2023-08-27-10-14.ckpt"))

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

    dataset_path = "roszcz/maestro-quantized"

    col1, col2 = st.columns(2)

    with col1:
        query_pitch = st.text_area(label="Query value for pitch embeddings", value="Liszt")
    with col2:
        query_velocity_time = st.text_area(label="Query value for velocity and time embeddings", value="Chopin")

    pitch_record, velocity_time_record = embeddings_based_on_query(
        pitch_encoder,
        velocity_time_encoder,
        dataset_path,
        query_pitch,
        query_velocity_time,
        device=device,
    )

    # st.write(pitch_record)
    # st.write(velocity_time_record)

    mp3_path = render_midi_to_mp3(pitch_record, velocity_time_record)

    st.audio(mp3_path, "audio/mp3")


if __name__ == "__main__":
    main()
