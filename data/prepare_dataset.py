import os

import fortepyan as ff
from tqdm import tqdm
from quantizer import MidiQuantizer
from datasets import Value, Dataset, Features, Sequence, DatasetDict, load_dataset


def process_dataset(dataset_path: str, split: str, sequence_len: int, quantizer: MidiQuantizer) -> list[dict]:
    dataset = load_dataset(dataset_path, split=split)

    processed_records = []

    for record in tqdm(dataset, total=dataset.num_rows):
        # print(record)
        piece = ff.MidiPiece.from_huggingface(record)
        processed_record = process_record(piece, sequence_len, quantizer)

        processed_records += processed_record

    return processed_records


def process_record(piece: ff.MidiPiece, sequence_len: int, quantizer: MidiQuantizer) -> list[dict]:
    piece_quantized = quantizer.quantize_piece(piece)

    midi_filename = piece_quantized.source["midi_filename"]

    record = []

    for subset in tqdm(piece_quantized.df.rolling(window=sequence_len, step=sequence_len)):
        # rolling sometimes creates subsets with shorter sequence length, they are filtered here
        if len(subset) != sequence_len:
            continue

        sequence = {
            "midi_filename": midi_filename,
            "pitch": subset.pitch.astype("int16").values.T,
            "dstart_bin": subset.dstart_bin.astype("int16").values.T,
            "duration_bin": subset.duration_bin.astype("int16").values.T,
            "velocity_bin": subset.velocity_bin.astype("int16").values.T,
        }

        record.append(sequence)

    return record


if __name__ == "__main__":
    # get huggingface token from environment variables
    token = os.environ["HUGGINGFACE_TOKEN"]

    # hyperparameters
    sequence_len = 128

    # That's where I'm downloading the LTAFDB data
    hf_dataset_path = "roszcz/maestro-v1"

    quantizer = MidiQuantizer(
        n_dstart_bins=7,
        n_duration_bins=7,
        n_velocity_bins=7,
    )

    train_records = process_dataset(hf_dataset_path, split="train", sequence_len=sequence_len, quantizer=quantizer)
    val_records = process_dataset(hf_dataset_path, split="validation", sequence_len=sequence_len, quantizer=quantizer)
    test_records = process_dataset(hf_dataset_path, split="test", sequence_len=sequence_len, quantizer=quantizer)

    # building huggingface dataset
    features = Features(
        {
            "midi_filename": Value(dtype="string"),
            "pitch": Sequence(feature=Value(dtype="int16"), length=sequence_len),
            "dstart_bin": Sequence(feature=Value(dtype="int16"), length=sequence_len),
            "duration_bin": Sequence(feature=Value(dtype="int16"), length=sequence_len),
            "velocity_bin": Sequence(feature=Value(dtype="int16"), length=sequence_len),
        }
    )

    # dataset = Dataset.from_list(records, features=features)
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_records, features=features),
            "validation": Dataset.from_list(val_records, features=features),
            "test": Dataset.from_list(test_records, features=features),
        }
    )

    # print(dataset["train"])
    dataset.push_to_hub("JasiekKaczmarczyk/maestro-quantized", token=token)
