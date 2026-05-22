from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from emulator_bench.common import ligand_cache_path, normalize_sequence, protein_cache_path, read_table, require_columns


class CachedMPEKDataset(Dataset):
    def __init__(
        self,
        table_path: str,
        embeddings_dir: str,
        sequence_col: str = "sequence",
        smiles_col: str = "smiles",
        target_col: str = "log10_value",
        target_mean: Optional[float] = None,
        target_std: Optional[float] = None,
        limit_rows: int = 0,
        validate_cache: bool = True,
        preload: bool = False,
    ):
        self.table_path = Path(table_path)
        self.embeddings_dir = Path(embeddings_dir)
        self.sequence_col = sequence_col
        self.smiles_col = smiles_col
        self.target_col = target_col
        frame = read_table(self.table_path)
        if limit_rows:
            frame = frame.head(limit_rows)
        require_columns(frame, [sequence_col, smiles_col, target_col], self.table_path)
        frame = frame[[sequence_col, smiles_col, target_col]].dropna(subset=[sequence_col, smiles_col, target_col]).reset_index(drop=True)
        self.sequences = [normalize_sequence(value) for value in frame[sequence_col].astype(str).tolist()]
        self.smiles = [str(value).strip() for value in frame[smiles_col].astype(str).tolist()]
        self.targets = frame[target_col].astype(float).to_numpy(dtype=np.float32)
        self.target_mean = float(np.mean(self.targets)) if target_mean is None else float(target_mean)
        std = float(np.std(self.targets)) if target_std is None else float(target_std)
        self.target_std = std if std > 1e-8 else 1.0
        self._preloaded = None

        if validate_cache:
            self._validate_cache()
        if preload:
            self._preloaded = [self._load_features(index) for index in range(len(self))]

    def _validate_cache(self) -> None:
        missing = []
        for sequence, smiles in zip(self.sequences, self.smiles):
            p_path = protein_cache_path(self.embeddings_dir, sequence)
            l_path = ligand_cache_path(self.embeddings_dir, smiles)
            if not p_path.exists():
                missing.append(str(p_path))
            if not l_path.exists():
                missing.append(str(l_path))
            if len(missing) >= 10:
                break
        if missing:
            raise FileNotFoundError("Missing cached embeddings. Run cache_embeddings.py first. Examples: " + "; ".join(missing))

    def _load_features(self, index: int):
        protein = np.load(protein_cache_path(self.embeddings_dir, self.sequences[index]), allow_pickle=True)["embedding"].astype(np.float32)
        ligand = np.load(ligand_cache_path(self.embeddings_dir, self.smiles[index]), allow_pickle=True)["embedding"].astype(np.float32)
        return protein, ligand

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        if self._preloaded is None:
            protein, ligand = self._load_features(index)
        else:
            protein, ligand = self._preloaded[index]
        raw_target = float(self.targets[index])
        norm_target = (raw_target - self.target_mean) / self.target_std
        return {
            "protein": torch.from_numpy(protein),
            "ligand": torch.from_numpy(ligand),
            "target": torch.tensor([norm_target], dtype=torch.float32),
            "raw_target": torch.tensor([raw_target], dtype=torch.float32),
        }
