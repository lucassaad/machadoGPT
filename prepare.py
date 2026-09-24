import os 
import pickle

import numpy as np

INPUT_DIR = 'inputs'
OUTPUT_DIR = 'inputs-bin'

class DataPreparer:

    def prepare(self, file: str):

        file_path = os.path.join(INPUT_DIR, file)


        file = file.split('.')[0]

        output_dir = os.path.join(os.path.dirname(__file__), OUTPUT_DIR)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, file)
        if not os.path.exists(output_path):
            os.makedirs(output_path)


        # get all the unique characters that occur in this text
        data, chars, vocab_size = self._get_vocab(file_path)
        print("all the unique characters:", ''.join(chars))
        print(f"vocab size: {vocab_size:,}")    

        # create a mapping from characters to integers
        stoi, itos = self._create_mapping(chars)

        # create the train and test splits
        train_data, val_data = self._split_data(data)

        # encode both to integers
        train_ids = self._encode(train_data, stoi)
        val_ids = self._encode(val_data, stoi)
        print(f"train has {len(train_ids):,} tokens")
        print(f"val has {len(val_ids):,} tokens")

        # export to bin files
        self._export_splits(train_ids, val_ids, output_path)

        # save the meta information as well, to help us encode/decode later
        self._save_metadata(vocab_size, itos, stoi, output_path)

        return chars, vocab_size, stoi, itos


    def _get_vocab(self, file: str):
        with open(file, 'r') as f:
            data = f.read()

        chars = sorted(list(set(data)))
        vocab_size = len(chars)

        return data, chars, vocab_size


    def _create_mapping(self, chars: list[str]): 
        stoi = { ch:i for i, ch in enumerate(chars) }
        itos = { i:ch for i, ch in enumerate(chars) }
        return stoi, itos


    def _encode(self, s: str, stoi: dict):
        return [stoi[c] for c in s]


    def _decode(self, l: list[int], itos: dict):
        return [itos[i] for i in l]


    def _split_data(self, data: str):
        n = len(data)
        train_data = data[:int(n*0.9)]
        val_data = data[int(n*0.9):]

        return train_data, val_data


    def _export_splits(self, train_ids, val_ids, output_path):
        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)
        train_ids.tofile(os.path.join(output_path, 'train.bin'))
        val_ids.tofile(os.path.join(output_path, 'val.bin'))


    def _save_metadata(self, vocab_size: int, itos: dict, stoi: dict, output_path):
        meta = {
            'vocab_size': vocab_size,
            'itos': itos,
            'stoi': stoi,
        }
        with open(os.path.join(output_path, 'meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)

   
if __name__ == "__main__":
    d = DataPreparer()
    d.prepare("texto.txt")

