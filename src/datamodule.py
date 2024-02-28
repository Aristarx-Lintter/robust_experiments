from typing import Optional

from pytorch_lightning import LightningDataModule
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from src.config import DataConfig
from src.efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater


class BirdViewDM(LightningDataModule):
    def __init__(
            self,
            config: DataConfig
    ):
        super().__init__()
        self.image_size = config.model_kwargs['input_sizes'][config.model_kwargs['compound_coef']]
        self._config = config

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            self.train_dataset = CocoDataset(
                root_dir=self._config.data_config.data_path, set=self._config.data_config.train_set,
                transform=transforms.Compose(
                    [Normalizer(mean=self._config.data_config.mean, std=self._config.data_config.std),
                     Augmenter(), Resizer(self.image_size)])
            )
            self.valid_dataset = CocoDataset(
                root_dir=self._config.data_config.data_path, set=self._config.data_config.val_set,
                transform=transforms.Compose(
                    [Normalizer(mean=self._config.data_config.mean, std=self._config.data_config.std),
                     Resizer(self.image_size)])
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, collate_fn=collater, **self._config.data_config.training_params.dict())

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_dataset, collate_fn=collater, **self._config.data_config.val_params.dict())
