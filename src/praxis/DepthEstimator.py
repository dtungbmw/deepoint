from transformers import GLPNImageProcessor, GLPNForDepthEstimation
from PIL import Image
import torch


class DepthEstimator:
    
    def predict(self):
        pass

    
class GLPNDepthEstimator(DepthEstimator):

    def __init__(self):
        self.feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
        self.model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

    def predict(self, image):
        #image = Image.open(image_file)
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
            return predicted_depth


class MidasDepthEstimator(DepthEstimator):

    def __init__(self):
        self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        self.midas.eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        #model_type = "DPT_Hybrid"
        #model_path = "model-f6b98070.pt"
        #midas_model = load_model(model_path, model_type)

    def predict(self, img):
        input_batch = self.transform(img).to("cuda" if torch.cuda.is_available() else "cpu")

        # Perform inference to get depth map
        with torch.no_grad():
            prediction = midas(input_batch)

        # Upsample the prediction to the original image size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        # Convert depth to numpy
        depth_map = prediction.cpu().numpy()
        return depth_map


    