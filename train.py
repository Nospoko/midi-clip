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
from datasets import load_dataset, concatenate_datasets

from data.dataset import MidiDataset
from models.pitch_encoder import PitchEncoder
from models.velocity_time_encoder import VelocityTimeEncoder

# from models.reverse_diffusion import Unet
# from ecg_segmentation_dataset import ECGDataset
# from models.forward_diffusion import ForwardDiffusion


def makedir_if_not_exists(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


def preprocess_dataset(dataset_name: list[str], batch_size: int, num_workers: int, *, overfit_single_batch: bool = False):
    hf_token = os.environ["HUGGINGFACE_TOKEN"]

    train_ds = []
    val_ds = []
    test_ds = []

    for ds_name in dataset_name:
        tr_ds = load_dataset(ds_name, split="train", use_auth_token=hf_token)
        v_ds = load_dataset(ds_name, split="validation", use_auth_token=hf_token)
        t_ds = load_dataset(ds_name, split="test", use_auth_token=hf_token)

        train_ds.append(tr_ds)
        val_ds.append(v_ds)
        test_ds.append(t_ds)

    train_ds = concatenate_datasets(train_ds)
    val_ds = concatenate_datasets(val_ds)
    test_ds = concatenate_datasets(test_ds)

    train_ds = MidiDataset(train_ds)
    val_ds = MidiDataset(val_ds)
    test_ds = MidiDataset(test_ds)

    if overfit_single_batch:
        train_ds = Subset(train_ds, indices=range(batch_size))
        val_ds = Subset(val_ds, indices=range(batch_size))
        test_ds = Subset(test_ds, indices=range(batch_size))

    # dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
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

    # accuracy
    acc_per_velocity_time = M.accuracy(logits, labels, task="multiclass", num_classes=batch_size)
    acc_per_pitch = M.accuracy(logits.t(), labels, task="multiclass", num_classes=batch_size)
    acc = (acc_per_velocity_time + acc_per_pitch) / 2

    # top k accuracy, k is 10% of batch size
    topk_acc_per_velocity_time = M.accuracy(
        logits, labels, task="multiclass", num_classes=batch_size, top_k=round(0.1 * batch_size)
    )
    topk_acc_per_pitch = M.accuracy(logits.t(), labels, task="multiclass", num_classes=batch_size, top_k=round(0.1 * batch_size))
    topk_acc = (topk_acc_per_velocity_time + topk_acc_per_pitch) / 2

    return loss, acc, topk_acc


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
    upload_file(
        path_or_fileobj=ckpt_save_path,
        path_in_repo=f"{cfg.logger.run_name}.ckpt",
        repo_id=cfg.paths.hf_repo_id,
        token=token,
    )


@torch.no_grad()
def validation_epoch(
    velocity_time_encoder: VelocityTimeEncoder,
    pitch_encoder: PitchEncoder,
    dataloader: DataLoader,
    cfg: OmegaConf,
    device: str,
) -> dict:
    pitch_encoder.eval()
    velocity_time_encoder.eval()
    val_loop = tqdm(enumerate(dataloader), total=len(dataloader))
    loss_epoch = 0.0
    acc_epoch = 0.0
    topk_acc_epoch = 0.0

    for batch_idx, batch in val_loop:
        # metrics returns loss and additional metrics if specified in step function
        loss, acc, topk_acc = forward_step(
            pitch_encoder, velocity_time_encoder, batch, temperature=cfg.train.temperature, device=device
        )

        val_loop.set_postfix(loss=loss.item())

        loss_epoch += loss.item()
        acc_epoch += acc
        topk_acc_epoch += topk_acc

    metrics = {
        "loss_epoch": loss_epoch / len(dataloader),
        "accuracy_epoch": acc_epoch / len(dataloader),
        "topk_accuracy_epoch": topk_acc_epoch / len(dataloader),
    }
    return metrics


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

    # validate on quantized maestro
    _, maestro_test, _ = preprocess_dataset(
        dataset_name="JasiekKaczmarczyk/maestro-quantized",
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
        loss_epoch = 0.0
        acc_epoch = 0.0
        topk_acc_epoch = 0.0

        for batch_idx, batch in train_loop:
            # metrics returns loss and additional metrics if specified in step function
            loss, acc, topk_acc = forward_step(
                pitch_encoder, velocity_time_encoder, batch, temperature=cfg.train.temperature, device=device
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loop.set_postfix(loss=loss.item())

            step_count += 1
            loss_epoch += loss.item()
            acc_epoch += acc
            topk_acc_epoch += topk_acc

            if (batch_idx + 1) % cfg.logger.log_every_n_steps == 0:
                # log metrics
                wandb.log({"train/loss": loss.item(), "train/accuracy": acc, "train/topk_accuracy": topk_acc}, step=step_count)

                # save model and optimizer states
                save_checkpoint(pitch_encoder, velocity_time_encoder, optimizer, cfg, save_path=save_path)

        training_metrics = {
            "train/loss_epoch": loss_epoch / len(train_dataloader),
            "train/accuracy_epoch": acc_epoch / len(train_dataloader),
            "train/topk_accuracy_epoch": topk_acc_epoch / len(train_dataloader),
        }

        # val epochs
        pitch_encoder.eval()
        velocity_time_encoder.eval()

        # Test on a split from a training dataset ...
        validation_metrics = validation_epoch(
            pitch_encoder=pitch_encoder,
            velocity_time_encoder=velocity_time_encoder,
            dataloader=val_dataloader,
            device=device,
            cfg=cfg,
        )
        validation_metrics = {"val/" + key: value for key, value in validation_metrics.items()}

        # ... and on maestro
        test_metrics = validation_epoch(
            pitch_encoder=pitch_encoder,
            velocity_time_encoder=velocity_time_encoder,
            dataloader=maestro_test,
            device=device,
            cfg=cfg,
        )
        test_metrics = {"maestro/" + key: value for key, value in test_metrics.items()}

        metrics = test_metrics | validation_metrics | training_metrics
        wandb.log(metrics, step=step_count)

    # save model at the end of training
    save_checkpoint(pitch_encoder, velocity_time_encoder, optimizer, cfg, save_path=save_path)

    wandb.finish()

    # upload model to huggingface if specified in cfg
    if cfg.paths.hf_repo_id is not None:
        upload_to_huggingface(save_path, cfg)


if __name__ == "__main__":
    wandb.login()

    train()
