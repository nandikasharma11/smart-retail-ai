import cv2
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
from typing import Tuple, List, Dict
from app.services import cv_utils

class FaceEmbedder:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load ResNet18 and use features up to global pooling layer
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.features.to(self.device)
        self.features.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        """Converts OpenCV BGR image cropped face to 512-dim embedding."""
        if face_bgr.size == 0:
            return np.zeros(512, dtype=np.float32)
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.features(tensor).squeeze().cpu().numpy()
        # L2 normalization
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

class FaceRecognizer:
    def __init__(self, db_path: str = '/Users/nandikasharma/smart-retail-ai/smart-retail-ai/app/models/face_db.pkl'):
        self.db_path = db_path
        self.embedder = FaceEmbedder()
        self.face_db = {}
        self.load_db()

    def load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    self.face_db = pickle.load(f)
                print(f"Face database loaded successfully from {self.db_path}. Registered names: {list(self.face_db.keys())}")
            except Exception as e:
                print(f"Error loading face database: {e}. Starting with an empty database.")
                self.face_db = {}
        else:
            self.face_db = {}

    def save_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.face_db, f)
        print(f"Face database saved to {self.db_path}")

    def register_face(self, name: str, image_bgr: np.ndarray) -> bool:
        """Detects face, extracts embedding, and registers in the database."""
        faces = cv_utils.detect_faces(image_bgr)
        if len(faces) == 0:
            return False
        
        # Take the largest face detected
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        face_crop = image_bgr[y:y+h, x:x+w]
        
        emb = self.embedder.get_embedding(face_crop)
        if name not in self.face_db:
            self.face_db[name] = []
        self.face_db[name].append(emb)
        self.save_db()
        return True

    def recognize_face(self, image_bgr: np.ndarray, threshold: float = 0.75) -> Tuple[str, float]:
        """Detects a face, computes similarity, and returns match name and confidence."""
        faces = cv_utils.detect_faces(image_bgr)
        if len(faces) == 0:
            return "No Face Detected", 0.0
        
        # Recognize the largest face detected
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        face_crop = image_bgr[y:y+h, x:x+w]
        
        query_emb = self.embedder.get_embedding(face_crop)
        
        best_name = "Unknown"
        best_sim = 0.0
        
        for name, embeddings in self.face_db.items():
            for db_emb in embeddings:
                # Cosine similarity is the dot product of two normalized vectors
                sim = float(np.dot(query_emb, db_emb))
                if sim > best_sim:
                    best_sim = sim
                    if sim >= threshold:
                        best_name = name
                        
        return best_name, best_sim

    def log_visit(self, name: str, log_path: str = '/Users/nandikasharma/smart-retail-ai/smart-retail-ai/data/customer_visits.csv') -> str:
        """Logs a customer visit with name and timestamp to a CSV file."""
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        file_exists = os.path.exists(log_path)
        with open(log_path, 'a') as f:
            if not file_exists:
                f.write('name,timestamp\n')
            f.write(f'{name},{timestamp}\n')
        return timestamp
