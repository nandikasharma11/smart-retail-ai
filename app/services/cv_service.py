import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple

class ProductClassifier:
    def __init__(self, model_path: str = '/Users/nandikasharma/smart-retail-ai/smart-retail-ai/app/models/product_classifier.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        self.model = mobilenet_v2()
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, len(self.classes))
        
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print("Product classifier loaded successfully.")
            except Exception as e:
                print(f"Error loading product classifier weights: {e}")
        else:
            print("Product classifier weights not found.")
            
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_bgr: np.ndarray) -> Tuple[str, float]:
        """Predicts product class and confidence score from OpenCV BGR image."""
        if image_bgr.size == 0:
            return "unknown", 0.0
        
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            
        return self.classes[pred_idx.item()], float(conf.item())
