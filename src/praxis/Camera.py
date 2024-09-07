import cv2


class Camera:
    
    def __init__(self):
        self.model = None

OUT_FILE = "/tmp/tmp2.jpg"
RANGE = 15
    
    
class MonocularCamera(Camera): 
    
    def __init__(self):
        pass

    def extractImage(self, video_file):
        cam = cv2.VideoCapture(video_file)
        for i in range(0, RANGE):
            ret, frame = cam.read()  # read 1st image
        cv2.imwrite(OUT_FILE, frame)
        cam.release()
        return OUT_FILE

        
    
    
    
    
    