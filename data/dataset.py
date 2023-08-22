import torch
import fortepyan as ff
from datasets import load_dataset
from quantizer import MidiQuantizer
from torch.utils.data import Dataset


class MidiDataset(Dataset):
    def __init__(self, huggingface_path: str, quantizer: MidiQuantizer, split: str = "train"):
        super().__init__()

        self.data = load_dataset(huggingface_path, split=split)
        self.quantizer = quantizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        piece = ff.MidiPiece.from_huggingface(self.data[index])

        quantized_piece = self.quantizer.quantize_piece(piece)

        # wrap signal and mask to torch.Tensor
        pitch = torch.tensor(quantized_piece.df.pitch)
        dstart_bin = torch.tensor(quantized_piece.df.dstart_bin)
        duration_bin = torch.tensor(quantized_piece.df.duration_bin)
        velocity_bin = torch.tensor(quantized_piece.df.velocity_bin)

        record = {
            "filename": quantized_piece.source["midi_filename"],
            "pitch": pitch,
            "dstart_bin": dstart_bin,
            "duration_bin": duration_bin,
            "velocity_bin": velocity_bin,
        }

        return record
