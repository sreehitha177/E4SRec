import csv
import os
import random
from datetime import datetime
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch


DEFAULT_COMPLETION_PATH = "data_preproc/user_sessions_with_completion.csv"
DEFAULT_SOURCE_PATH = "data_preproc/user_sessions_lastfm1k_minuser1000_minitem7_sessgap1200_minsesslen10_minhist50.csv"
DEFAULT_MASTER_MAP_PATH = "datasets/sequential/LastFM/item_id_master_map.csv"
MIN_SEQ_LEN = 5
TEST_QUANTILE = 0.9
VAL_QUANTILE_PRETEST = 0.9
VAL_USER_RATIO = 0.1
RANDOM_SEED = 42


def _parse_timestamp(value):
    if value.endswith("Z"):
        value = value[:-1]
    return datetime.fromisoformat(value)


def _quantile_datetime(values, q):
    ts = pd.to_datetime(pd.Series(values))
    return ts.quantile(q).to_pydatetime()


def _load_completion_splits(
    source_path,
    completion_path,
    master_map_path,
    train_split,
    val_split,
    test_split,
):
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Missing source interaction file: {source_path}")
    if not os.path.exists(completion_path):
        raise FileNotFoundError(f"Missing completion-ratio file: {completion_path}")
    if not os.path.exists(master_map_path):
        raise FileNotFoundError(f"Missing item master map: {master_map_path}")

    track_to_item = {}
    with open(master_map_path, newline="", encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            track_to_item[row["track_id"]] = int(row["item_id"])

    user_counts = {}
    all_timestamps = []
    with open(source_path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 7:
                continue
            user_id, ts, _, _, track_id, _, _ = row[:7]
            if track_id not in track_to_item:
                continue
            ts_dt = _parse_timestamp(ts)
            all_timestamps.append(ts_dt)
            user_counts[user_id] = user_counts.get(user_id, 0) + 1

    test_cutoff = _quantile_datetime(all_timestamps, TEST_QUANTILE)
    pretest_timestamps = [ts for ts in all_timestamps if ts <= test_cutoff]
    val_cutoff = _quantile_datetime(pretest_timestamps, VAL_QUANTILE_PRETEST)

    candidate_users = [uid for uid in sorted(user_counts) if user_counts[uid] >= MIN_SEQ_LEN]
    rng = random.Random(RANDOM_SEED)
    val_user_count = max(1, int(round(len(candidate_users) * VAL_USER_RATIO)))
    val_users = set(rng.sample(candidate_users, val_user_count))

    ratio_splits = {}
    new_uid = 1

    def finalize_user(uid, records):
        nonlocal new_uid
        if uid is None or user_counts.get(uid, 0) < MIN_SEQ_LEN:
            return
        records = sorted(records, key=lambda rec: rec[0])
        is_val_user = uid in val_users
        if is_val_user:
            preliminary_train = [rec for rec in records if rec[0] <= val_cutoff]
            preliminary_test = [rec for rec in records if rec[0] > test_cutoff]
        else:
            preliminary_train = [rec for rec in records if rec[0] <= test_cutoff]
            preliminary_test = [rec for rec in records if rec[0] > test_cutoff]
        if len(preliminary_train) < 1 or len(preliminary_test) < 1:
            return

        new_user = new_uid - 1
        expected_full = (
            train_split.get(new_user, [])
            + val_split.get(new_user, [])
            + test_split.get(new_user, [])
        )
        if len(expected_full) != len(records):
            raise ValueError(
                f"Completion/source length mismatch for user {new_uid}: "
                f"{len(records)} records vs {len(expected_full)} split items"
            )

        aligned_records = []
        cursor = 0
        idx = 0
        while idx < len(records):
            ts_value = records[idx][0]
            group = []
            while idx < len(records) and records[idx][0] == ts_value:
                group.append(records[idx])
                idx += 1

            expected_group = expected_full[cursor:cursor + len(group)]
            grouped_by_item = {}
            for rec in group:
                grouped_by_item.setdefault(rec[1], []).append(rec)
            if sorted([rec[1] for rec in group]) != sorted(expected_group):
                raise ValueError(f"Timestamp-batch alignment failed for user {new_uid}")
            for item in expected_group:
                if item not in grouped_by_item or len(grouped_by_item[item]) == 0:
                    raise ValueError(f"Missing item {item} in timestamp batch for user {new_uid}")
                aligned_records.append(grouped_by_item[item].pop(0))
            cursor += len(group)

        records = aligned_records

        if is_val_user:
            train_records = [rec for rec in records if rec[0] <= val_cutoff]
            val_records = [rec for rec in records if val_cutoff < rec[0] <= test_cutoff]
            test_records = [rec for rec in records if rec[0] > test_cutoff]
        else:
            train_records = [rec for rec in records if rec[0] <= test_cutoff]
            val_records = []
            test_records = [rec for rec in records if rec[0] > test_cutoff]

        if len(train_records) < 1 or len(test_records) < 1:
            return

        generated_train = [item for _, item, _ in train_records]
        generated_val = [item for _, item, _ in val_records]
        generated_test = [item for _, item, _ in test_records]
        if generated_train != train_split.get(new_user, []):
            raise ValueError(f"Completion alignment mismatch for train split user {new_uid}")
        if generated_val != val_split.get(new_user, []):
            raise ValueError(f"Completion alignment mismatch for val split user {new_uid}")
        if generated_test != test_split.get(new_user, []):
            raise ValueError(f"Completion alignment mismatch for test split user {new_uid}")

        ratio_splits[new_user] = {
            "train": [ratio for _, _, ratio in train_records],
            "val": [ratio for _, _, ratio in val_records],
            "test": [ratio for _, _, ratio in test_records],
        }
        new_uid += 1

    current_user = None
    current_records = []
    with open(source_path, newline="", encoding="utf-8", errors="ignore") as fs, open(
        completion_path, newline="", encoding="utf-8", errors="ignore"
    ) as fc:
        source_reader = csv.reader(fs)
        completion_reader = csv.DictReader(fc)
        for source_row, completion_row in zip(source_reader, completion_reader):
            if len(source_row) < 7:
                continue

            user_id, ts, _, _, track_id, _, _ = source_row[:7]
            if track_id not in track_to_item:
                continue

            completion_key = (
                completion_row["user_id"],
                completion_row["timestamp"],
                completion_row["session_id"],
                completion_row["artist_name"],
                completion_row["track_name"],
            )
            source_key = (
                source_row[0],
                source_row[1],
                source_row[6],
                source_row[3],
                source_row[5],
            )
            if completion_key != source_key:
                raise ValueError(
                    f"Completion/source row mismatch: {source_key} != {completion_key}"
                )

            if current_user is not None and user_id != current_user:
                finalize_user(current_user, current_records)
                current_records = []
            current_user = user_id
            ratio_value = completion_row.get("completion_ratio", "")
            current_records.append(
                (
                    _parse_timestamp(ts),
                    track_to_item[track_id],
                    float(ratio_value) if ratio_value else 0.0,
                )
            )

    finalize_user(current_user, current_records)

    return ratio_splits


class BipartiteGraphDataset(Dataset):
    def __init__(self, dataset):
        super(BipartiteGraphDataset, self).__init__()
        self.dataset = dataset

        self.trainData, self.allPos, self.testData = [], {}, {}
        self.n_user, self.m_item = 0, 0
        with open(self.dataset + 'train.txt', 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                user, items = int(line[0]), [int(item) + 1 for item in line[1:]]
                self.allPos[user] = items
                for item in items:
                    self.trainData.append([user, item])
                self.n_user = max(self.n_user, user)
                self.m_item = max(self.m_item, max(items))

        with open(self.dataset + 'test.txt', 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                user, items = int(line[0]), [int(item) + 1 for item in line[1:]]
                self.testData[user] = items
                self.n_user = max(self.n_user, user)
                self.m_item = max(self.m_item, max(items))

        self.n_user, self.m_item = self.n_user + 1, self.m_item + 1

    def __getitem__(self, idx):
        user, item = self.trainData[idx]
        return user, self.allPos[user], item

    def __len__(self):
        return len(self.trainData)


@dataclass
class BipartiteGraphCollator:
    def __call__(self, batch) -> dict:
        user, items, labels = zip(*batch)
        bs = len(user)
        max_len = max([len(item) for item in items])
        inputs = [[user[i]] + items[i] + [0] * (max_len - len(items[i])) for i in range(bs)]
        inputs_mask = [[1] + [1] * len(items[i]) + [0] * (max_len - len(items[i])) for i in range(bs)]
        labels = [[label] for label in labels]
        inputs, inputs_mask, labels = torch.LongTensor(inputs), torch.LongTensor(inputs_mask), torch.LongTensor(labels)

        return {
            "inputs": inputs,
            "inputs_mask": inputs_mask,
            "labels": labels
        }


class SequentialDataset(Dataset):
    def __init__(
        self,
        dataset,
        maxlen,
        use_completion_ratio=False,
        completion_path=None,
        source_path=None,
        master_map_path=None,
    ):
        super(SequentialDataset, self).__init__()
        self.dataset = dataset
        self.maxlen = maxlen
        self.use_completion_ratio = use_completion_ratio
        self.completion_path = completion_path or DEFAULT_COMPLETION_PATH
        self.source_path = source_path or DEFAULT_SOURCE_PATH
        self.master_map_path = master_map_path or DEFAULT_MASTER_MAP_PATH

        self.trainData, self.valData, self.testData = [], {}, {}
        self.n_user, self.m_item = 0, 0

        def _read_split_file(path):
            split = {}
            with open(path, "r") as f:
                for line in f:
                    parts = line.strip().split(" ")
                    if len(parts) < 2:
                        continue
                    user = int(parts[0]) - 1
                    items = [int(item) for item in parts[1:]]
                    split[user] = items
            return split

        train_split = _read_split_file(self.dataset + "train.txt")
        test_split = _read_split_file(self.dataset + "test.txt")
        val_split = _read_split_file(self.dataset + "val.txt")
        test_sample_split = _read_split_file(self.dataset + "test_sample.txt")
        ratio_splits = None
        if self.use_completion_ratio:
            ratio_splits = _load_completion_splits(
                source_path=self.source_path,
                completion_path=self.completion_path,
                master_map_path=self.master_map_path,
                train_split=train_split,
                val_split=val_split,
                test_split=test_split,
            )

        self.allPos = {}
        all_users = sorted(set(train_split.keys()) | set(test_split.keys()))
        for user in all_users:
            train_items = train_split.get(user, [])
            test_items = test_split.get(user, [])
            if len(train_items) < 1 or len(test_items) < 1:
                continue

            sample_items = test_sample_split.get(user)
            if sample_items is None or len(sample_items) < 1:
                sample_items = [test_items[0]]
            self.allPos[user] = sample_items
            train_ratios = None
            val_ratios = None
            if self.use_completion_ratio:
                user_ratios = ratio_splits.get(user)
                if user_ratios is None:
                    train_ratios = [0.0] * len(train_items)
                    val_ratios = [0.0] * len(val_split.get(user, []))
                else:
                    train_ratios = user_ratios["train"]
                    val_ratios = user_ratios["val"]

            if self.use_completion_ratio:
                self.testData[user] = [train_items, train_ratios, test_items[0]]
            else:
                self.testData[user] = [train_items, test_items[0]]
            if user in val_split and len(val_split[user]) > 0:
                if self.use_completion_ratio:
                    self.valData[user] = [train_items, train_ratios, val_split[user][0]]
                else:
                    self.valData[user] = [train_items, val_split[user][0]]
            else:
                self.valData[user] = []

            length = min(len(train_items), self.maxlen)
            for t in range(length):
                seq_items = train_items[:-length + t]
                label = train_items[-length + t]
                if self.use_completion_ratio:
                    seq_ratios = train_ratios[:-length + t]
                    self.trainData.append([seq_items, seq_ratios, label])
                else:
                    self.trainData.append([seq_items, label])

            self.n_user = max(self.n_user, user)
            local_max = max(
                max(train_items),
                max(test_items),
                max(self.allPos[user]),
                max(val_split[user]) if user in val_split and len(val_split[user]) > 0 else 0,
            )
            self.m_item = max(self.m_item, local_max)

        self.n_user, self.m_item = self.n_user + 1, self.m_item + 1

    def get_user_pos_items(self, users):
        posItems = []
        for user in users:
            posItems.append(self.allPos[user])
        return posItems

    def __getitem__(self, idx):
        if self.use_completion_ratio:
            seq, completion_ratio, label = self.trainData[idx]
            return seq, completion_ratio, label
        seq, label = self.trainData[idx]
        return seq, label

    def __len__(self):
        return len(self.trainData)

    def get_eval_record(self, user, subset="test"):
        store = self.testData if subset == "test" else self.valData
        record = store.get(user, [])
        if not record:
            return [], None, None
        if self.use_completion_ratio:
            return record[0], record[1], record[2]
        return record[0], None, record[1]

@dataclass
class SequentialCollator:
    def __call__(self, batch) -> dict:
        has_completion_ratio = len(batch[0]) == 3
        if has_completion_ratio:
            seqs, completion_ratios, labels = zip(*batch)
        else:
            seqs, labels = zip(*batch)
        max_len = max(max([len(seq) for seq in seqs]), 2)
        inputs = [[0] * (max_len - len(seq)) + seq for seq in seqs]
        inputs_mask = [[0] * (max_len - len(seq)) + [1] * len(seq) for seq in seqs]
        labels = [[label] for label in labels]
        batch_dict = {
            "inputs": torch.LongTensor(inputs),
            "inputs_mask": torch.LongTensor(inputs_mask),
            "labels": torch.LongTensor(labels),
        }
        if has_completion_ratio:
            completion_ratio = [
                [0.0] * (max_len - len(ratios)) + ratios for ratios in completion_ratios
            ]
            batch_dict["completion_ratio"] = torch.FloatTensor(completion_ratio)

        return batch_dict
