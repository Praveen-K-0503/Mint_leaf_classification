"""
Mint Leaf AI — Image Quality Pre-Screening Module
Evaluates image sharpness (Laplacian variance) and brightness exposure prior to AI inference.
"""

import cv2
import numpy as np
from PIL import Image

def check_image_quality(pil_image, blur_threshold=100.0, min_brightness=30.0, max_brightness=235.0):
    """
    Evaluates PIL Image for sharpness and exposure quality.
    
    Returns:
        dict: {
            "is_valid": bool,
            "blur_score": float,
            "brightness_score": float,
            "quality_status": str,
            "message": str
        }
    """
    # Convert PIL Image to OpenCV BGR / Gray
    cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # 1. Variance of Laplacian for Sharpness
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # 2. Mean Brightness for Exposure
    brightness_score = float(np.mean(gray))
    
    is_valid = True
    status = "EXCELLENT"
    message = "Image quality is optimal for AI diagnostic analysis."
    
    if blur_score < blur_threshold:
        is_valid = False
        status = "DEFOCUS_BLUR_DETECTED"
        message = f"Image is out of focus (Sharpness score: {blur_score:.1f} < {blur_threshold:.1f}). Please capture a sharp, focused leaf image."
    elif brightness_score < min_brightness:
        is_valid = False
        status = "UNDEREXPOSED_DARK"
        message = f"Image is too dark (Brightness score: {brightness_score:.1f} < {min_brightness:.1f}). Please increase lighting or disable canopy shadow."
    elif brightness_score > max_brightness:
        is_valid = False
        status = "OVEREXPOSED_GLARE"
        message = f"Image is overexposed with daylight glare (Brightness score: {brightness_score:.1f} > {max_brightness:.1f}). Please avoid direct harsh sunlight."
        
    return {
        "is_valid": is_valid,
        "blur_score": round(blur_score, 2),
        "brightness_score": round(brightness_score, 2),
        "quality_status": status,
        "message": message
    }
