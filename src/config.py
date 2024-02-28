from typing import List

from omegaconf import OmegaConf
from pydantic import BaseModel


class LossConfig(BaseModel):
    name: str
    loss_fn: str
    loss_kwargs: dict


class SetParams(BaseModel):
    batch_size: int
    shuffle: bool
    drop_last: bool
    num_workers: int


class DataConfig(BaseModel):
    data_path: str
    train_set: str
    val_set: str
    n_workers: int
    mean: List[float]
    std: List[float]
    anchors_scales: str
    anchors_ratios: str
    obj_list: List[str]
    training_params: SetParams
    val_params: SetParams


class Config(BaseModel):
    project_name: str
    experiment_name: str
    data_config: DataConfig
    n_epochs: int
    accelerator: str
    device: int
    monitor_metric: str
    monitor_mode: str
    model: str
    model_kwargs: dict
    optimizer: str
    optimizer_kwargs: dict
    scheduler: str
    scheduler_kwargs: dict
    loss: str
    loss_kwargs: dict

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
