import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class MidiDataset(Dataset):
    def __init__(self, huggingface_path: str, split: str = "train"):
        super().__init__()

        self.data = load_dataset(huggingface_path, split=split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.data[index]

        # wrap signal and mask to torch.Tensor
        pitch = torch.tensor(sequence["pitch"], dtype=torch.long)
        dstart_bin = torch.tensor(sequence["dstart_bin"], dtype=torch.long)
        duration_bin = torch.tensor(sequence["duration_bin"], dtype=torch.long)
        velocity_bin = torch.tensor(sequence["velocity_bin"], dtype=torch.long)

        record = {
            "filename": sequence["midi_filename"],
            "pitch": pitch,
            "dstart_bin": dstart_bin,
            "duration_bin": duration_bin,
            "velocity_bin": velocity_bin,
        }

        return record
