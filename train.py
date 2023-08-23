import os
import math

import hydra
import torch
import wandb
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from omegaconf import OmegaConf
import torchmetrics.functional as M
from huggingface_hub import upload_file
from torch.utils.data import Subset, DataLoader

from data.dataset import MidiDataset
from models.pitch_encoder import PitchEncoder
from models.velocity_time_encoder import VelocityTimeEncoder

# from models.reverse_diffusion import Unet
# from ecg_segmentation_dataset import ECGDataset
# from models.forward_diffusion import ForwardDiffusion


def makedir_if_not_exists(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


def preprocess_dataset(dataset_name: str, batch_size: int, num_workers: int, *, overfit_single_batch: bool = False):
    train_ds = MidiDataset(dataset_name, split="train")
    val_ds = MidiDataset(dataset_name, split="validation")
    test_ds = MidiDataset(dataset_name, split="test")

    if overfit_single_batch:
        train_ds = Subset(train_ds, indices=range(batch_size))
        val_ds = Subset(val_ds, indices=range(batch_size))
        test_ds = Subset(test_ds, indices=range(batch_size))

    # dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader


def forward_step(
    pitch_encoder: PitchEncoder,
    velocity_time_encoder: VelocityTimeEncoder,
    batch: dict[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    temperature: float,
    device: torch.device,
) -> tuple[float, float]:
    pitch = batch["pitch"].to(device)
    dstart_bin = batch["dstart_bin"].to(device)
    duration_bin = batch["duration_bin"].to(device)
    velocity_bin = batch["velocity_bin"].to(device)

    batch_size = pitch.shape[0]

    # shape: [batch_size, embedding_dim]
    pitch_embeddings = pitch_encoder(pitch)

    # shape: [batch_size, embedding_dim]
    velocity_time_embeddings = velocity_time_encoder(velocity_bin, dstart_bin, duration_bin)

    # normalization
    pitch_embeddings = pitch_embeddings / torch.norm(pitch_embeddings, p=2, dim=1, keepdim=True)
    velocity_time_embeddings = velocity_time_embeddings / torch.norm(velocity_time_embeddings, p=2, dim=1, keepdim=True)

    # scaled pairwise cosine similarities, out shape: [logits_per_pitch, logits_per_velocity_time]
    logits = torch.einsum("n e, m e -> n m", [pitch_embeddings, velocity_time_embeddings])
    logits = logits * math.exp(temperature)

    labels = torch.arange(0, batch_size, dtype=torch.long, device=device)

    loss_per_velocity_time = F.cross_entropy(logits, labels)
    loss_per_pitch = F.cross_entropy(logits.t(), labels)

    # get loss value for batch
    loss = (loss_per_velocity_time + loss_per_pitch) / 2

    # other metrics
    acc_per_velocity_time = M.accuracy(logits, labels, task="multiclass", num_classes=batch_size)
    acc_per_pitch = M.accuracy(logits.t(), labels, task="multiclass", num_classes=batch_size)

    acc = (acc_per_velocity_time + acc_per_pitch) / 2

    return loss, acc


def save_checkpoint(
    pitch_encoder: PitchEncoder,
    velocity_time_encoder: VelocityTimeEncoder,
    optimizer: optim.Optimizer,
    cfg: OmegaConf,
    save_path: str,
):
    # saving models
    torch.save(
        {
            "pitch_encoder": pitch_encoder.state_dict(),
            "velocity_time_encoder": velocity_time_encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        },
        f=save_path,
    )


def upload_to_huggingface(ckpt_save_path: str, cfg: OmegaConf):
    # get huggingface token from environment variables
    token = os.environ["HUGGINGFACE_TOKEN"]

    # upload model to hugging face
    upload_file(ckpt_save_path, path_in_repo=f"{cfg.logger.run_name}.ckpt", repo_id=cfg.paths.hf_repo_id, token=token)


@hydra.main(config_path="configs", config_name="config-default", version_base="1.3.2")
def train(cfg: OmegaConf):
    # create dir if they don't exist
    makedir_if_not_exists(cfg.paths.log_dir)
    makedir_if_not_exists(cfg.paths.save_ckpt_dir)

    # dataset
    train_dataloader, val_dataloader, _ = preprocess_dataset(
        dataset_name=cfg.train.dataset_name,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        overfit_single_batch=cfg.train.overfit_single_batch,
    )

    # logger
    wandb.init(project="midi-clip", name=cfg.logger.run_name, dir=cfg.paths.log_dir)

    device = torch.device(cfg.train.device)

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

    # setting up optimizer
    optimizer = optim.AdamW(
        params=list(pitch_encoder.parameters()) + list(velocity_time_encoder.parameters()),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    # load checkpoint if specified in cfg
    if cfg.paths.load_ckpt_path is not None:
        checkpoint = torch.load(cfg.paths.load_ckpt_path)

        pitch_encoder.load_state_dict(checkpoint["pitch_encoder"])
        velocity_time_encoder.load_state_dict(checkpoint["velocity_time_encoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    # ckpt specifies directory and name of the file is name of the experiment in wandb
    save_path = f"{cfg.paths.save_ckpt_dir}/{cfg.logger.run_name}.ckpt"

    # step counts for logging to wandb
    step_count = 0

    for epoch in range(cfg.train.num_epochs):
        # train epoch
        pitch_encoder.train()
        velocity_time_encoder.train()
        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

        for batch_idx, batch in train_loop:
            # metrics returns loss and additional metrics if specified in step function
            loss, acc = forward_step(
                pitch_encoder, velocity_time_encoder, batch, temperature=cfg.train.temperature, device=device
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loop.set_postfix(loss=loss.item())

            step_count += 1

            if (batch_idx + 1) % cfg.logger.log_every_n_steps == 0:
                # log metrics
                wandb.log({"train/loss": loss.item(), "train/accuracy": acc}, step=step_count)

                # save model and optimizer states
                save_checkpoint(pitch_encoder, velocity_time_encoder, optimizer, cfg, save_path=save_path)

        # val epoch
        pitch_encoder.eval()
        velocity_time_encoder.eval()
        val_loop = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        loss_epoch = 0.0
        acc_epoch = 0.0

        with torch.no_grad():
            for batch_idx, batch in val_loop:
                # metrics returns loss and additional metrics if specified in step function
                loss, acc = forward_step(
                    pitch_encoder, velocity_time_encoder, batch, temperature=cfg.train.temperature, device=device
                )

                val_loop.set_postfix(loss=loss.item())

                loss_epoch += loss.item()
                acc_epoch += acc

            wandb.log(
                {"val/loss_epoch": loss_epoch / len(val_dataloader), "val/accuracy_epoch": acc_epoch / len(val_dataloader)},
                step=step_count,
            )

    # save model at the end of training
    save_checkpoint(pitch_encoder, velocity_time_encoder, optimizer, cfg, save_path=save_path)

    wandb.finish()

    # upload model to huggingface if specified in cfg
    if cfg.paths.hf_repo_id is not None:
        upload_to_huggingface(save_path, cfg)


if __name__ == "__main__":
    wandb.login()

    train()
