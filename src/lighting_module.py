from dataclasses import dataclass

import torch
from torch import nn
import pytorch_lightning as pl

from src.backbone import get_model
from src.config import Config
from src.helpers.loader import load_object


@dataclass
class Loss:
    name: str
    loss: nn.Module


class BirdViewVehicleModule(pl.LightningModule):
    def __init__(
            self,
            config: Config
    ):
        super().__init__()

        self._config = config
        compound_coef = config.model_kwargs['compound_coef']
        num_classes = len(config.data_config.obj_list)
        self.model = get_model(
            num_classes=num_classes, compound_coef=compound_coef,
            ratios=eval(config.data_config.anchors_ratios),
            scales=eval(config.data_config.anchors_scales)
        )
        self._loss = Loss(name=config.loss.split('.')[-1], loss=load_object(config.loss)(**config.loss_kwargs))

    def forward(self, imgs: torch.Tensor):
        return self.model(imgs)

    def configure_optimizers(self):
        optimizer = load_object(self._config.optimizer)(
            self.model.parameters(),
            **self._config.optimizer_kwargs,
        )
        scheduler = load_object(self._config.scheduler)(optimizer, **self._config.scheduler_kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_total_loss',
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    def training_step(self, batch, batch_idx):
        """
        Считаем лосс.
        """
        images, annotations = batch['img'], batch['annot']
        _, regression, classification, anchors = self(images)
        return self._calculate_loss(classification, regression, anchors, annotations, 'train_')

    def validation_step(self, batch, batch_idx):
        """
        Считаем лосс и метрики.
        """
        images, annotations = batch['img'], batch['annot']
        _, regression, classification, anchors = self(images)
        self._calculate_loss(classification, regression, anchors, annotations, 'val_')

    def test_step(self, batch, batch_idx):
        """
        Считаем метрики.
        """
        images, annotations = batch['img'], batch['annot']
        _, regression, classification, anchors = self(images)
        self._calculate_loss(classification, regression, anchors, annotations, 'test_')

    def _calculate_loss(
            self,
            classification,
            regression,
            anchors,
            annotations,
            prefix: str,
    ) -> torch.Tensor:
        cls_loss, reg_loss = self._loss.loss(classification, regression, anchors, annotations)
        self.log(f'{prefix}{self._loss.name}_cls_loss', cls_loss.item())
        self.log(f'{prefix}{self._loss.name}_reg_loss', reg_loss.item())
        total_loss = cls_loss + reg_loss
        self.log(f'{prefix}total_loss', total_loss.item())
        return total_loss
