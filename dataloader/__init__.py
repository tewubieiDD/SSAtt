from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset


class DomainDataset(Dataset):
    def __init__(self, x, y, d):
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))
        self.d = torch.from_numpy(np.asarray(d, dtype=np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return {"x": self.x[index], "d": self.d[index]}, self.y[index]


def load_mat_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    mat = loadmat(path, squeeze_me=False, struct_as_record=False)
    x = np.asarray(mat["x"], dtype=np.float32)
    y = np.asarray(mat["y"]).reshape(-1).astype(np.int64)
    subject = np.asarray(mat["subject"]).reshape(-1).astype(np.int64)
    session = np.asarray(mat["session"]).reshape(-1).astype(np.int64)
    d_session = np.asarray(mat["domain_inter_session"]).reshape(-1).astype(np.int64)
    d_subject = np.asarray(mat["domain_inter_subject"]).reshape(-1).astype(np.int64)
    return {"x": x, "y": y, "subject": subject, "session": session,
            "domain_inter_session": d_session, "domain_inter_subject": d_subject,
            "file": str(path)}


def merge_records(records):
    if not records:
        raise ValueError("records cannot be empty")
    keys = ("x", "y", "subject", "session", "domain_inter_session", "domain_inter_subject")
    return {key: np.concatenate([record[key] for record in records], axis=0) for key in keys}


def make_dataloader(record, indices, domain_key, batch_size=64, shuffle=False,
                    add_channel_dim=True, num_workers=0, pin_memory=False):
    """Create a loader from an explicit record and explicit sample indices."""
    indices = np.asarray(indices, dtype=np.int64)
    x = record["x"][indices]
    if add_channel_dim and x.ndim == 3:
        x = x[:, None, :, :]
    y = record["y"][indices]
    d = record[domain_key][indices]
    return DataLoader(
        DomainDataset(x, y, d),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
