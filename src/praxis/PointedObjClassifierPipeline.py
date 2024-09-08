import numpy as np
from torch.utils.data import DataLoader
import torch
import dataset
from tqdm import tqdm
from model import build_pointing_network
from omegaconf import DictConfig, OmegaConf

from praxis.DepthEstimator import *
from praxis.GesturePointingDirectionEstimator import *
from praxis.ObjectDetector import YOLOWorldObjectDetector
from praxis.DepthEstimator import GLPNDepthEstimator
from praxis.Camera import MonocularCamera
from praxis.Utilities import *

    
class PointedObjClassifierPipeline:
    
    def __init__(self):
        pass

    
    def classify(self, dl, cfg, DEVICE):
        res_class = None
        res_p = 0
        ## step 1. camera
        camera = MonocularCamera()
        image_file = camera.extractImage(cfg.movie)

        objectDetector = YOLOWorldObjectDetector()
        objectDetectorResults = objectDetector.predict_and_vis(image_file)

        deepointointingEstimator = DeepointointingEstimator()
        network = deepointointingEstimator.build_network(cfg, DEVICE)

        gLPNDepthEstimator = GLPNDepthEstimator()
        depth_map = gLPNDepthEstimator.predict(image_file)
        iter =0
        for batch in tqdm(dl):
            iter = iter + 1
            if iter < 23:
                continue
            result = network(batch)
            # bs may be smaller than cfg.hardware.bs for the last iteration
            bs = batch["abs_joint_position"].shape[0]
            for i_bs in range(bs):
                joints = batch["abs_joint_position"][i_bs][-1].to("cpu").numpy()
                #image = batch["orig_image"][i_bs].to("cpu").numpy() / 255

                direction = result["direction"][i_bs]
                #print(f'direction={direction}')
                prob_pointing = float(
                    (result["action"][i_bs, 1].exp() / result["action"][i_bs].exp().sum())
                )
                #print(f"{prob_pointing=}")
                if prob_pointing >= 0.8:
                    print(f"******$$$$$$$$$$ {prob_pointing=}")
                    hand_idx = 9 if batch["lr"][i_bs] == "l" else 10
                    calculate_intersection(joints[hand_idx], direction, objectDetectorResults[0].boxes, depth_map)
                    exit(0)
                else:
                    print(f"|||||||$$$$$$$$$$ {prob_pointing=}")
                #
                #ORIG_HEIGHT, ORIG_WIDTH = image.shape[:2]

                #arrow_base = (
                ##        joints[hand_idx] / np.array((ORIG_WIDTH, ORIG_HEIGHT)) * 2 - 1
                #)
                #print(f">>>>>>>> i_bs = {i_bs}")
                #print(f">>>>>>>> ORIG_HEIGHT, ORIG_WIDTH= {ORIG_HEIGHT}, {ORIG_WIDTH}")
                #print(f">>>>>>>> joints[hand_idx]= {joints[hand_idx]}")
                #print(f">>>>>>>> Arrow_base= {arrow_base}")
        return res_class, res_p
    
    
    
    
    
    
    