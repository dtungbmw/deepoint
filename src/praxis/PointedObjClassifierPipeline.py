from praxis.DepthEstimator import *
from praxis.ObjectDetector import ObjectDetector
from praxis.GesturePointingDirectionEstimator import *
from praxis.Camera import MonocularCamera
from torch.utils.data import DataLoader
import torch
import dataset
from model import build_pointing_network
from omegaconf import DictConfig, OmegaConf

    
class PointedObjClassifierPipeline:
    
    def __init__(self, pointingDirEstimator, objectDetector):
        self.pointingDirEstimator = pointingDirEstimator
        self.objectDetector = objectDetector

    def detect_objects(self, video):
        results = self.objectDetector.predict(video)
        self.objectDetector.print_results(results)
        self.objectDetector.visualize(results)
        return results
    
    def classify(self, video):
        res_class = None
        res_p = 0
        ##
        detection_results = self.detect_objects(video)

        tlength= 15
        nworkers = 8
        bs = 32
        lr = 'r'
        movie = video
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        # iterate video
        ds = dataset.MovieDataset(movie, lr, tlength, DEVICE)
        print("create dl")
        dl = DataLoader(
            ds,
            batch_size=bs,
            num_workers=nworkers,
        )
        
        
        
        return res_class, res_p
    
    
    
    
    
    
    