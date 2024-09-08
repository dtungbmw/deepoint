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

    Path("demo").mkdir(exist_ok=True)
    '''
    fps = 15
    out_green = cv2.VideoWriter(
        f"demo/{Path(cfg.movie).name}-processed-green-{cfg.lr}.mp4",
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        fps,
        (WIDTH, HEIGHT),
    )
    out_greenred = cv2.VideoWriter(
        f"demo/{Path(cfg.movie).name}-processed-greenred-{cfg.lr}.mp4",
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        fps,
        (WIDTH, HEIGHT),
    )
    prev_arrow_base = np.array((0, 0))
    '''
    '''
    for batch in tqdm(dl):
        result = network(batch)
        # bs may be smaller than cfg.hardware.bs for the last iteration
        bs = batch["abs_joint_position"].shape[0]
        for i_bs in range(bs):
            joints = batch["abs_joint_position"][i_bs][-1].to("cpu").numpy()
            image = batch["orig_image"][i_bs].to("cpu").numpy() / 255

            direction = result["direction"][i_bs]
            print(f'direction={direction}')
            prob_pointing = float(
                (result["action"][i_bs, 1].exp() / result["action"][i_bs].exp().sum())
            )
            print(f"{prob_pointing=}")

            ##
            calculate_intersection(None, None, objectDetectorResults, None)
    
    
            ORIG_HEIGHT, ORIG_WIDTH = image.shape[:2]
            hand_idx = 9 if batch["lr"][i_bs] == "l" else 10
            if (joints[hand_idx] < 0).any():
                arrow_base = prev_arrow_base
            else:
                arrow_base = (
                    joints[hand_idx] / np.array((ORIG_WIDTH, ORIG_HEIGHT)) * 2 - 1
                )
                prev_arrow_base = arrow_base

            image_green = draw_arrow_on_image(
                image,
                (
                    arrow_base[0],
                    -arrow_base[1],
                    direction[0].cpu(),
                    direction[2].cpu(),
                    -direction[1].cpu(),
                ),
                dict(
                    acolor=(
                        0,
                        1,
                        0,
                    ),  # Green. OpenCV uses BGR
                    asize=0.05 * prob_pointing,
                    offset=0.02,
                ),
            )
            image_greenred = draw_arrow_on_image(
                image,
                (
                    arrow_base[0],
                    -arrow_base[1],
                    direction[0].cpu(),
                    direction[2].cpu(),
                    -direction[1].cpu(),
                ),
                dict(
                    acolor=(
                        0,
                        prob_pointing,
                        1 - prob_pointing,
                    ),  # Green to red. OpenCV uses BGR
                    asize=0.05 * prob_pointing,
                    offset=0.02,
                ),
            )

            cv2.imshow("", image_green)
            cv2.waitKey(10)

            out_green.write((image_green * 255).astype(np.uint8))
            out_greenred.write((image_greenred * 255).astype(np.uint8))
    '''
    return



if __name__ == "__main__":
    main()
