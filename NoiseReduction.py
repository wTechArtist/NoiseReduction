import torch
import cv2
import numpy as np


class NoiseReduction:
    
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.001,
                    # "display": "slider"
                }),
                "preserve_details": ("FLOAT", {
                    "default": 100.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    # "display": "slider"
                }),
                "reduce_noise": ("FLOAT", {
                    "default": 100.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    # "display": "slider"
                }),
                "sharpen_details": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    # "display": "slider"
                }),
                "remove_jpeg_artifact": (["enable", "disable"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "image/filters"

    def execute(self, image, strength, preserve_details, reduce_noise, sharpen_details, remove_jpeg_artifact):
        # Convert from torch tensor [B,H,W,C] to numpy array
        image_np = image.numpy()
        
        # Scale to 0-255 range and convert to uint8
        image_np = (image_np * 255).astype(np.uint8)
        
        # Process each image in batch
        processed_images = []
        for img in image_np:
            # Apply noise reduction
            h = strength * 3  # Convert strength to h parameter
            template_window = int(preserve_details / 10) * 2 + 3  # Convert preserve_details to odd number
            search_window = int(reduce_noise / 10) * 2 + 21  # Convert reduce_noise to odd number
            
            denoised = cv2.fastNlMeansDenoisingColored(img, 
                                                      None,
                                                      h=h,
                                                      hColor=h,
                                                      templateWindowSize=template_window,
                                                      searchWindowSize=search_window)
            
            # Apply sharpening if sharpen_details > 0
            if sharpen_details > 0:
                blur = cv2.GaussianBlur(denoised, (0, 0), 3)
                sharp = cv2.addWeighted(denoised, 1.5, blur, -0.5, 0)
                denoised = cv2.addWeighted(denoised, 1 - sharpen_details/100, 
                                         sharp, sharpen_details/100, 0)
            
            # Remove JPEG artifacts if enabled
            if remove_jpeg_artifact == "enable":
                denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
                
            processed_images.append(denoised)
            
        # Convert back to torch tensor and normalize to 0-1
        processed_tensor = torch.from_numpy(np.stack(processed_images).astype(np.float32) / 255.0)
        
        return (processed_tensor,)

