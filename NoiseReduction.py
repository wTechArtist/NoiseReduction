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
                    "default": 0.3,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "去噪强度。0=无去噪，0.1-0.5=轻微去噪保留细节，0.5-1.0=中等去噪，1.0-2.0=强力去噪"
                }),
                "preserve_details": ("FLOAT", {
                    "default": 70.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "tooltip": "细节保留程度。数值越高，色彩去噪越温和，保留更多颜色细节。0=强力色彩去噪，100=完全保留色彩"
                }),
                "search_window": ("INT", {
                    "default": 35,
                    "min": 7,
                    "max": 35,
                    "step": 2,
                    "tooltip": "搜索窗口大小。较大值能更好去除噪声但计算较慢。7-15=快速处理，21-25=平衡效果，25-35=最佳效果"
                }),
                "template_window": ("INT", {
                    "default": 7,
                    "min": 3,
                    "max": 11,
                    "step": 2,
                    "tooltip": "模板窗口大小。用于计算像素权重的邻域大小。3-5=保留锐利边缘，7=平衡，9-11=更平滑结果"
                }),
                "blend_with_original": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "与原图混合比例。0=纯去噪结果，0.2-0.4=保留部分原始细节，0.5+=更多原始纹理"
                }),
                "sharpen_details": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "tooltip": "锐化强度。0=无锐化，10-30=轻微锐化，40-70=中等锐化，80-100=强锐化。过高可能产生伪影"
                }),
                "remove_jpeg_artifact": (["disable", "enable"], {
                    "tooltip": "移除JPEG压缩伪影。enable=使用双边滤波去除块状伪影，适用于压缩过的图像"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "image/filters"

    def execute(self, image, strength, preserve_details, search_window, template_window, 
                blend_with_original, sharpen_details, remove_jpeg_artifact):
        # 如果强度为0，直接返回原图
        if strength <= 0.0:
            return (image,)
        
        # Convert from torch tensor [B,H,W,C] to numpy array
        image_np = image.numpy()
        
        # Scale to 0-255 range and convert to uint8
        image_np = (image_np * 255).astype(np.uint8)
        
        # Process each image in batch
        processed_images = []
        for img in image_np:
            # 保存原图用于后续混合
            original_img = img.copy()
            
            # 更精细的强度控制 - 使用非线性映射
            # strength=0.1 -> h≈1, strength=1.0 -> h≈10, strength=2.0 -> h≈20
            if strength <= 1.0:
                h = strength * 10.0  # 0-10范围，更精细控制
            else:
                h = 10.0 + (strength - 1.0) * 10.0  # 10-20范围，强力去噪
            
            # 确保窗口大小为奇数
            template_win = template_window if template_window % 2 == 1 else template_window + 1
            search_win = search_window if search_window % 2 == 1 else search_window + 1
            
            # Apply noise reduction
            denoised = cv2.fastNlMeansDenoisingColored(img, 
                                                      None,
                                                      h=h,
                                                      hColor=h * (preserve_details / 100.0),  # 根据preserve_details调整色彩去噪强度
                                                      templateWindowSize=template_win,
                                                      searchWindowSize=search_win)
            
            # 与原图混合以保留细节
            if blend_with_original > 0.0:
                denoised = cv2.addWeighted(original_img, blend_with_original, 
                                         denoised, 1.0 - blend_with_original, 0)
            
            # Apply sharpening if sharpen_details > 0
            if sharpen_details > 0:
                # 创建锐化核
                kernel = np.array([[-1,-1,-1],
                                 [-1, 9,-1],
                                 [-1,-1,-1]]) * (sharpen_details / 100.0)
                kernel[1,1] = 8 + (sharpen_details / 100.0)
                
                # 应用锐化
                sharpened = cv2.filter2D(denoised, -1, kernel)
                # 混合锐化结果
                denoised = cv2.addWeighted(denoised, 1 - sharpen_details/100, 
                                         sharpened, sharpen_details/100, 0)
            
            # Remove JPEG artifacts if enabled
            if remove_jpeg_artifact == "enable":
                # 使用更温和的双边滤波参数
                denoised = cv2.bilateralFilter(denoised, 5, 50, 50)
                
            processed_images.append(denoised)
            
        # Convert back to torch tensor and normalize to 0-1
        processed_tensor = torch.from_numpy(np.stack(processed_images).astype(np.float32) / 255.0)
        
        return (processed_tensor,)

