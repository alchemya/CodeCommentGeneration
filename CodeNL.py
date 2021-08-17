from torchtext.data.datasets_utils import (
    _RawTextIterableDataset,
    _read_text_iterator,
)

NUM_LINES = {
    'train': 400000,
    'valid': 3000,
    'test': 1000,
}

DATASET_NAME = "CodeNL"


def CodeNL(split, src_path, trg_path):
    src_data_iter = _read_text_iterator(src_path)
    trg_data_iter = _read_text_iterator(trg_path)

    return _RawTextIterableDataset(DATASET_NAME, NUM_LINES[split], zip(src_data_iter, trg_data_iter))

