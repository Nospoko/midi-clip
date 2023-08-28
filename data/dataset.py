import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset as HFDataset


class MidiDataset(Dataset):
    def __init__(self, dataset: HFDataset):
        super().__init__()

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.dataset[index]

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

        # I have different version locally than in HF
        if "source" in sequence.keys():
            record.update({"source": sequence["source"]})

        return record
