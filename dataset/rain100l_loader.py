import os
import re
import cv2
import numpy as np
from torch.utils.data import Dataset


class Rain100LDataset(Dataset):
    def __init__(self, rainy_dir, clean_dir, transform=None):
        self.rainy_dir = rainy_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.image_pairs = self._build_image_pairs()

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        rainy_path, clean_path = self.image_pairs[idx]
        rainy = cv2.imread(rainy_path)
        clean = cv2.imread(clean_path)
        rainy = cv2.cvtColor(rainy, cv2.COLOR_BGR2RGB)
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        # Resize
        rainy = cv2.resize(rainy, (256, 256))
        clean = cv2.resize(clean, (256, 256))
        # Normalize to [0,1]
        rainy = rainy.astype(np.float32) / 255.0
        clean = clean.astype(np.float32) / 255.0
        if self.transform:
            rainy = self.transform(rainy)
            clean = self.transform(clean)
        return rainy, clean

    def _build_image_pairs(self):
        rainy_images = self._map_images_by_id(self.rainy_dir)
        clean_images = self._map_images_by_id(self.clean_dir)
        common_ids = sorted(set(rainy_images) & set(clean_images))

        if not common_ids:
            raise ValueError("No matching rainy/clean image pairs were found.")

        missing_rainy = sorted(set(clean_images) - set(rainy_images))
        missing_clean = sorted(set(rainy_images) - set(clean_images))
        if missing_rainy or missing_clean:
            raise ValueError(
                f"Mismatch in image pairs. Missing rainy ids: {missing_rainy[:5]}, "
                f"missing clean ids: {missing_clean[:5]}"
            )

        return [(rainy_images[image_id], clean_images[image_id]) for image_id in common_ids]

    def _map_images_by_id(self, directory):
        image_map = {}
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            image_id = self._extract_image_id(name)
            if image_id is None:
                continue
            image_map[image_id] = path
        return image_map

    @staticmethod
    def _extract_image_id(filename):
        match = re.search(r"norain-(\d+)", filename)
        if match is None:
            return None
        return int(match.group(1))
