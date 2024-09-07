from ultralytics import YOLOWorld
from praxis.Camera import MonocularCamera

class ObjectDetector:
    
    def __init__(self):
        self.model = None
    
    def predict(self):
        pass

    def print_results(self, results):
        for result in results:
            print("box-> ")
            print(result.boxes)
            print(result.probs)
    
class YOLOWorldObjectDetector(ObjectDetector):
    
    def __init__(self):
        self.model = YOLOWorld("yolov8s-world.pt")  # or select yolov8m/l-world.pt for different sizes
        
    def predict(self, video):
        monocularCamera = MonocularCamera()
        image = monocularCamera.extractImage(video)
        self.image = image
        self.model.set_classes(["laptop" , "lamp", "fan"])
        results = self.model.predict(image)    
        return results

    def predict_and_vis(self, video):
        results = self.predict(video)
        self.print_results(results)
        self.visualize(results)
        return results
    
    def visualize(self, results):
        results[0].show()
    

            
    
    
    
    
    
    