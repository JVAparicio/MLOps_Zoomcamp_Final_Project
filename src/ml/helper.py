import os
from datetime import datetime, timedelta
from hydra import compose, initialize
from omegaconf import DictConfig
import pickle


def load_config() -> DictConfig:
    with initialize(version_base=None, config_path="./config"):
        config = compose(config_name="main")
    return config


def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)