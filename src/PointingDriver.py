import numpy as np
import cv2
import torch
import dataset
from pathlib import Path

import hydra
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from model import build_pointing_network
#from draw_arrow import WIDTH, HEIGHT
from praxis.ObjectDetector import YOLOWorldObjectDetector
from praxis.GesturePointingDirectionEstimator import DeepointointingEstimator
from praxis.PointedObjClassifierPipeline import PointedObjClassifierPipeline
from praxis.Utilities import *

@hydra.main(version_base=None, config_path="../conf", config_name="base")
def main(cfg: DictConfig) -> None:
    import logging

    logging.info(
        "Successfully loaded settings:\n"
        + "==================================================\n"
        + f"{OmegaConf.to_yaml(cfg)}"
        + "==================================================\n"
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cpu":
        logging.warning("Running DeePoint with CPU takes a long time.")

    assert (
        cfg.movie is not None
    ), "Please specify movie path as `movie=/path/to/movie.mp4`"

    assert (
        cfg.lr is not None
    ), "Please specify whether the pointing hand is left or right with `lr=l` or `lr=r`."

    assert cfg.ckpt is not None, "checkpoint should be specified for evaluation"

    cfg.hardware.bs = 2
    cfg.hardware.nworkers = 0
    ds = dataset.MovieDataset(cfg.movie, cfg.lr, cfg.model.tlength, DEVICE)
    dl = DataLoader(
        ds,
        batch_size=cfg.hardware.bs,
        num_workers=cfg.hardware.nworkers,
    )

    pointedObjClassifierPipeline = PointedObjClassifierPipeline()
    pointedObjClassifierPipeline.classify(dl, cfg, DEVICE)

    #Path("demo").mkdir(exist_ok=True)

    return



if __name__ == "__main__":
    main()
