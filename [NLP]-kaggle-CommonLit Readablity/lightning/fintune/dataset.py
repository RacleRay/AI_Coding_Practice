import numpy as np
import multiprocessing
import more_itertools

import torch
from torch.utils.data import Sampler, Dataset, DataLoader
from transformers import AutoTokenizer
from config import CONFIG


class BERTDataset(Dataset):
    def __init__(self, df):
        self.text = df['excerpt'].values
        self.target = df['target'].values
        self.max_len = CONFIG.max_len
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG.tokenizer)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]
        text = ' '.join(text.split())
        inputs = self.tokenizer.encode_plus(text,
                                            None,
                                            truncation=True,
                                            add_special_tokens=True,
                                            max_length=self.max_len,
                                            padding='max_length',
                                            return_token_type_ids=True)

        return {
            'input_ids':
                torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask':
                torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids':
                torch.tensor(inputs["token_type_ids"], dtype=torch.long),
            'label':
                torch.tensor(self.target[index], dtype=torch.float)
        }


class RoBERTaDataset(Dataset):
    def __init__(self, df, for_test=False):
        self.text = df['excerpt'].values
        self.for_test = for_test
        if not for_test:
            self.target = df['target'].values
        self.max_len = CONFIG.max_len
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG.tokenizer)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]
        text = ' '.join(text.split())
        inputs = self.tokenizer.encode_plus(text,
                                            None,
                                            truncation=True,
                                            add_special_tokens=True,
                                            max_length=self.max_len,
                                            padding='max_length',
                                            return_token_type_ids=True)

        if not self.for_test:
            return {
                'input_ids':
                    torch.tensor(inputs['input_ids'], dtype=torch.long),
                'attention_mask':
                    torch.tensor(inputs['attention_mask'], dtype=torch.long),
                'label':
                    torch.tensor(self.target[index], dtype=torch.float)
            }
        else:
            return {
                'input_ids':
                    torch.tensor(inputs['input_ids'], dtype=torch.long),
                'attention_mask':
                    torch.tensor(inputs['attention_mask'], dtype=torch.long)
            }


class SmartBatchingDataset(Dataset):
    "tokenize并得到dataloader, 得到roberta的输入, 和BERT相比，没有token_type_ids"
    def __init__(self, df):
        super(SmartBatchingDataset, self).__init__()
        tokenizer = AutoTokenizer.from_pretrained(CONFIG.tokenizer)
        # 这里 df.excerpt 表示dataframe中的文本所在列，使用其他dataframe时需要替换
        self._data = (
            f"{tokenizer.bos_token} " + df.excerpt + f" {tokenizer.eos_token}"
        	).apply(tokenizer.tokenize).apply(tokenizer.convert_tokens_to_ids).to_list()
        self._targets = None
        if 'target' in df.columns:
            self._targets = df.target.tolist()
        self.sampler = None

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        if self._targets is not None:
            return self._data[item], self._targets[item]
        else:
            return self._data[item]

    def get_dataloader(self, batch_size, max_len, pad_id):
        self.sampler = SmartBatchingSampler(
            data_source=self._data,
            batch_size=batch_size
        )
        collate_fn = SmartBatchingCollate(
            targets=self._targets,
            max_length=max_len,
            pad_token_id=pad_id
        )
        dataloader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            sampler=self.sampler,
            collate_fn=collate_fn,
            num_workers=(multiprocessing.cpu_count()-1),
            pin_memory=True
        )
        return dataloader


class SmartBatchingSampler(Sampler):
    "按序列长度排序，得到一组shuffle之后的batch data"
    def __init__(self, data_source, batch_size):
        super(SmartBatchingSampler, self).__init__(data_source)
        self.len = len(data_source)
        sample_lengths = [len(seq) for seq in data_source]
        argsort_inds = np.argsort(sample_lengths)
        self.batches = list(more_itertools.chunked(argsort_inds, n=batch_size))
        self._backsort_inds = None

    def __iter__(self):
        if self.batches:
            last_batch = self.batches.pop(-1)
            np.random.shuffle(self.batches)
            self.batches.append(last_batch)
        self._inds = list(more_itertools.flatten(self.batches))
        yield from self._inds

    def __len__(self):
        return self.len

    @property
    def backsort_inds(self):
        "未shuffle时，batch序列按照长度排序的结果"
        if self._backsort_inds is None:
            self._backsort_inds = np.argsort(self._inds)
        return self._backsort_inds

class SmartBatchingCollate:
    "每个batch分别pad到最大长度，得到attention mask，处理target"
    def __init__(self, targets, max_length, pad_token_id):
        self._targets = targets
        self._max_length = max_length
        self._pad_token_id = pad_token_id

    def __call__(self, batch):
        if self._targets is not None:
            sequences, targets = list(zip(*batch))
        else:
            sequences = list(batch)

        input_ids, attention_mask = self.pad_sequence(
            sequences,
            max_sequence_length=self._max_length,
            pad_token_id=self._pad_token_id
        )

        if self._targets is not None:
            # output = input_ids, attention_mask, torch.tensor(targets)
            output = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': torch.tensor(targets, dtype=torch.float)
            }
        else:
            # output = input_ids, attention_mask
            output = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
        return output

    def pad_sequence(self, sequence_batch, max_sequence_length, pad_token_id):
        max_batch_len = max(len(sequence) for sequence in sequence_batch)
        max_len = min(max_batch_len, max_sequence_length)
        padded_sequences, attention_masks = [[] for i in range(2)]
        attend, no_attend = 1, 0
        for sequence in sequence_batch:
            # 限制model所允许的最大长度
            new_sequence = list(sequence[:max_len])

            attention_mask = [attend] * len(new_sequence)
            pad_length = max_len - len(new_sequence)

            new_sequence.extend([pad_token_id] * pad_length)
            attention_mask.extend([no_attend] * pad_length)

            padded_sequences.append(new_sequence)
            attention_masks.append(attention_mask)

        padded_sequences = torch.tensor(padded_sequences)
        attention_masks = torch.tensor(attention_masks)
        return padded_sequences, attention_masks