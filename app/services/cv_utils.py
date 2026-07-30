import cv2
import numpy as np
import os
from typing import List, Tuple

def read_image(image_path: str) -> np.ndarray:
    """Reads an image from path. Raises FileNotFoundError if path doesn't exist."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not decode image at: {image_path}")
    return image

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts a BGR or RGB image to grayscale."""
    if len(image.shape) == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def resize_image(image: np.ndarray, width: int = None, height: int = None, keep_aspect: bool = True) -> np.ndarray:
    """Resizes an image. Keeps aspect ratio if keep_aspect is True and only width or height is provided."""
    h, w = image.shape[:2]
    if width is None and height is None:
        return image.copy()
    
    if keep_aspect:
        if width is not None and height is None:
            ratio = width / float(w)
            dim = (width, int(h * ratio))
        elif height is not None and width is None:
            ratio = height / float(h)
            dim = (int(w * ratio), height)
        else:
            # Both provided, but keeping aspect ratio: scale to fit within dimensions
            ratio = min(width / float(w), height / float(h))
            dim = (int(w * ratio), int(h * ratio))
    else:
        dim = (width if width is not None else w, height if height is not None else h)
        
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def blur_image(image: np.ndarray, kernel_size: int = 5, method: str = 'gaussian') -> np.ndarray:
    """Applies blur to the image. Methods: 'gaussian', 'median', 'bilateral'."""
    if kernel_size % 2 == 0:
        kernel_size += 1  # Kernel size must be odd
    
    if method.lower() == 'gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method.lower() == 'median':
        return cv2.medianBlur(image, kernel_size)
    elif method.lower() == 'bilateral':
        # For bilateral blur, kernel_size is used as diameter of pixel neighborhood
        return cv2.bilateralFilter(image, kernel_size, 75, 75)
    else:
        raise ValueError(f"Unknown blur method: {method}")

def detect_edges(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """Applies Canny edge detection."""
    return cv2.Canny(image, low_threshold, high_threshold)

def detect_faces(image: np.ndarray, scale_factor: float = 1.1, min_neighbors: int = 5) -> List[Tuple[int, int, int, int]]:
    """Detects faces using OpenCV's Haar Cascade. Returns a list of bounding boxes (x, y, w, h)."""
    gray = to_grayscale(image)
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise IOError(f"Failed to load Haar cascade from: {cascade_path}")
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(30, 30))
    # Convert numpy array of faces to list of tuples
    return [tuple(map(int, face)) for face in faces]

def draw_face_boxes(image: np.ndarray, faces: List[Tuple[int, int, int, int]], color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Draws bounding boxes around detected faces on a copy of the image."""
    img_copy = image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, thickness)
    return img_copy
