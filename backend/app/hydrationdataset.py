import os
import pandas as pd
import torch
from torch.utils.data import Dataset


LABEL_MAP = {
    "dehydrated": 0,
    "hydrated": 1,
    "mildly": 2
}

class HydrationDataset(Dataset):
    def __init__(self, data_dir, window_size=150, step_size=50):
        self.samples = []
        global_index = 0

        for fname in os.listdir(data_dir):
            if fname.endswith(".csv"):
                df = pd.read_csv(os.path.join(data_dir, fname))

                X = df[["gsr_raw", "temp_raw", "hr_raw"]].values
                y = LABEL_MAP[str(df["label"].iloc[0]).strip().lower()]

                for start in range(0, len(X) - window_size, step_size):
                    window = X[start:start+window_size]

                    self.samples.append((
                        torch.tensor(window, dtype=torch.float32),
                        torch.tensor(y, dtype=torch.long),
                        global_index
                    ))
                    global_index += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]