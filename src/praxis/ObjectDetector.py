from ultralytics import YOLOWorld


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
        
    def predict(self, image):
        self.model.set_classes(["tv", "laptop", "lamp", "fan"])
        results = self.model.predict(image)    
        return results

    def predict_and_vis(self, image):
        results = self.predict(image)
        self.print_results(results)
        self.visualize(results)
        return results
    
    def visualize(self, results):
        results[0].show()
    

            
    
    
    
    
    
    