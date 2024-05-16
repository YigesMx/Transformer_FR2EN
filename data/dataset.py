import os
import pickle
from typing import Optional

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import datasets

class TranslationDataset(Dataset):
    def __init__(self, split, wrapped_tokenizer, with_raw_text: Optional[bool] = None):
        assert split in ["train", "validation", "test"], "split must be one of 'train', 'validation', 'test'"

        # self.dataset = datasets.load_dataset("wmt/wmt14", "fr-en", split=split)
        self.dataset = datasets.load_dataset("iwslt2017", "iwslt2017-en-fr", split=split, trust_remote_code=True)
        
        assert wrapped_tokenizer is not None
        self.warped_tokenizer = wrapped_tokenizer
        
        if with_raw_text is None:
            if split in ["train", "validation"]:
                self.with_raw_text = False
            elif split in ["test"]:
                self.with_raw_text = True
        else:
            self.with_raw_text = with_raw_text

        self.src_fr, self.tgt_en = [], []

        if os.path.exists(f"data/cache/{split}_src_fr.pkl") and os.path.exists(f"data/cache/{split}_tgt_en.pkl"):
            print(f"Loading {split} dataset into memory from cache")
            with open(f"data/cache/{split}_src_fr.pkl", "rb") as f:
                self.src_fr = pickle.load(f)
            with open(f"data/cache/{split}_tgt_en.pkl", "rb") as f:
                self.tgt_en = pickle.load(f)
            print(f"Loaded {len(self.src_fr)} samples")

        else:
            for data in tqdm(self.dataset, desc=f"Tokenizing {split} dataset into memory"):
                self.src_fr.append(
                    self.warped_tokenizer.tokenize_src(
                        data['translation']['fr']
                    )[0]
                )
                self.tgt_en.append(
                    self.warped_tokenizer.tokenize_tgt(
                        data['translation']['en']
                    )[0]
                )
                assert (int(len(self.src_fr[-1])) <= int(self.warped_tokenizer.max_len - 2)), \
                        f"Tokenized text is too long: {len(self.src_fr[-1])} > {self.warped_tokenizer.max_len - 2}"
            print(f"Tokenized {len(self.src_fr)} samples")

            # save into data cache as .pkl
            print(f"Saving {split} dataset into cache...")
            with open(f"data/cache/{split}_src_fr.pkl", "wb") as f:
                pickle.dump(self.src_fr, f)
            with open(f"data/cache/{split}_tgt_en.pkl", "wb") as f:
                pickle.dump(self.tgt_en, f)
            print(f"Saved {split} dataset into cache")
        
        assert len(self.src_fr) == len(self.tgt_en) == len(self.dataset) and len(self.src_fr) > 0


    def __len__(self):
        return len(self.dataset)#//20

    def __getitem__(self, idx):
        #idx *= 20
        if self.with_raw_text:
            return self.dataset[idx]['translation']['fr'], self.dataset[idx]['translation']['en'], self.src_fr[idx], self.tgt_en[idx]
        else:
            return self.src_fr[idx], self.tgt_en[idx]
        # src_text = self.dataset[idx]['translation']['fr']
        # tgt_text = self.dataset[idx]['translation']['en']

        # if self.warped_tokenizer is None:
        #     return src_text, tgt_text
        # else:
        #     src = self.warped_tokenizer.tokenize_src(src_text)
        #     input_tgt, output_tgt = self.warped_tokenizer.tokenize_tgt(tgt_text)
        #     return src[0], input_tgt[0], output_tgt[0]
    
    def collate_fn(self, batch):
        if self.with_raw_text:
            src_text_batch, tgt_text_batch, src_batch, tgt_batch = zip(*batch)
            src_batch = self.warped_tokenizer.pad_batch(src_batch)
            tgt_batch = self.warped_tokenizer.pad_batch(tgt_batch)
            return src_text_batch, tgt_text_batch, src_batch, tgt_batch
        else:
            src_batch, tgt_batch = zip(*batch)
            src_batch = self.warped_tokenizer.pad_batch(src_batch)
            tgt_batch = self.warped_tokenizer.pad_batch(tgt_batch)
            return src_batch, tgt_batch

def get_dataloader(split, batch_size=64, shuffle=False, wrapped_tokenizer=None):
    dataset = TranslationDataset(split, wrapped_tokenizer)
    return DataLoader(
            dataset, 
            batch_size=batch_size, 
            collate_fn=dataset.collate_fn,
            num_workers=8,
            shuffle=shuffle
        )
