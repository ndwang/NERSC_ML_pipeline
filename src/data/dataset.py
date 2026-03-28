import torch
from torch.utils.data import Dataset
import numpy as np

class FrequencyMapDataset(Dataset):
    def __init__(self, maps_path, scales_path, centroids_path=None, transform=None):
        self.maps_path = maps_path
        self.scales_path = scales_path
        self.centroids_path = centroids_path
        self.transform = transform

        # Open in mmap mode 'r' (read-only)
        self.maps = np.load(maps_path, mmap_mode='r')
        self.scales = np.load(scales_path, mmap_mode='r')

        if centroids_path is not None:
            self.centroids = np.load(centroids_path, mmap_mode='r')
        else:
            # Default to zeros if no centroids file (backward compat)
            self.centroids = np.zeros((len(self.maps), 6), dtype=np.float32)

        if len(self.maps) != len(self.scales):
            raise ValueError(
                f"maps ({len(self.maps)}) and scales ({len(self.scales)}) "
                "must have the same number of samples"
            )

    def __len__(self):
        return self.maps.shape[0]

    def __getitem__(self, idx):
        maps = torch.from_numpy(self.maps[idx].copy()).float()
        scales = torch.from_numpy(self.scales[idx].copy()).float()
        centroids = torch.from_numpy(self.centroids[idx].copy()).float()

        if self.transform:
            maps = self.transform(maps)

        return maps, scales, centroids
