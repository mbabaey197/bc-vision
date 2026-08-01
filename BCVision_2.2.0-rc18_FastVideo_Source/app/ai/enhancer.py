import cv2

def enhance(image):
    if image is None:
        return None
    return cv2.detailEnhance(image) if hasattr(cv2,"detailEnhance") else image
