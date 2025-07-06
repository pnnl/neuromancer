from dataclasses import dataclass
from typing import Self
import neuromancer as nm
from numpy.typing import NDArray
import torch


@dataclass(frozen=True, slots=True)
class DataSource:
    data: nm.dataset.DictDataset
    loader: torch.utils.data.DataLoader

    @classmethod
    def init(
        cls,
        raw_data: dict[str, NDArray],
        rows: slice,
        name: str,
        dtype: torch.dtype = torch.float32,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> Self:
        assert all(v.ndim == 2 for v in raw_data.values())
        assert len(set(v.shape[0] for v in raw_data.values())) == 1
        data = nm.dataset.DictDataset(
            {
                key: torch.tensor(value[rows, :], dtype=dtype)
                for key, value in raw_data.items()
            },
            name=name,
        )
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            collate_fn=data.collate_fn,
            shuffle=shuffle,
        )
        return cls(data=data, loader=loader)
