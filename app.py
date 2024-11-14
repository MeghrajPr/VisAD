from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
import torch
from torch import nn
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI()

# Load pre-trained ResNet feature extractor and autoencoder model
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.eval()
        
        # Disable gradient calculation for ResNet parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Hook for intermediate layers
        def hook(module, input, output):
            self.features.append(output)
            
        # Register hooks
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

    def forward(self, x):
        self.features = []
        with torch.no_grad():
            _ = self.model(x)
        fmap_size = self.features[0].shape[-2]
        resize = nn.AdaptiveAvgPool2d(fmap_size)
        resized_maps = [resize(fmap) for fmap in self.features]
        patch = torch.cat(resized_maps, dim=1)
        return patch

# Initialize models
feature_extractor = ResNetFeatureExtractor()
autoencoder = torch.load("autoencoder_model.pth")  # Path to your trained autoencoder model
autoencoder.eval()

# Transformation pipeline for input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust as per your model requirements
    transforms.ToTensor(),
])

# Decision function to calculate anomaly score
def decision_function(segm_map):
    mean_top10_values = []
    for map in segm_map:
        flatten_t = map.view(-1)
        sorted_tensor, _ = torch.sort(flatten_t, descending=True)
        mean_top10_values.append(sorted_tensor[:10].mean())
    return torch.stack(mean_top10_values)

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=415, detail="Unsupported file type.")
    
    # Load image and transform
    image = Image.open(io.BytesIO(await file.read()))
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        # Extract features using ResNet
        features = feature_extractor(image)
        
        # Pass through autoencoder
        recon = autoencoder(features)
        
        # Calculate reconstruction error
        seg_map = ((features - recon) ** 2).mean(dim=1)[:, 3:-3, 3:-3]
        anomaly_score = decision_function(seg_map)
        
        # Check if anomaly score exceeds threshold
        if anomaly_score.item() > 0.018:
            result = "Not Okay"
        else:
            result = "Okay"

    return JSONResponse({"result": result})

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "Anomaly detection model is running."}
