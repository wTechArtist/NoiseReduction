import torch
import torch.nn.functional as F
import cv2
import numpy as np


class DepthToNormals:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "depth": ("IMAGE", {
                    "tooltip": "输入的深度图像，深度值将被转换为法线贴图"
                }),
                "scale": ("FLOAT",{"default": 0.1, "min": 0.001, "max": 1000, "step": 0.001,
                                  "tooltip": "深度缩放系数，控制法线的强度和凸凹程度。值越大，表面起伏越明显"}),
                "output_mode": (["Standard", "BAE", "MiDaS"], {
                    "tooltip": "法线贴图输出模式：Standard=标准模式，BAE=翻转X轴，MiDaS=翻转XY轴顺序"
                }),
                "denoise_method": (["none", "gentle_blur", "nlmeans_light", "bilateral_light"], {
                    "tooltip": "深度去噪方法：none=不去噪，gentle_blur=轻度高斯模糊，nlmeans_light=轻度非局部均值，bilateral_light=轻度双边滤波"
                }),
                "denoise_strength": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 2.0, "step": 0.05,
                    "tooltip": "去噪强度。建议0.1-0.5用于轻度去噪，保持细节"}),
                "detail_preservation": ("FLOAT", {"default": 0.95, "min": 0.7, "max": 1.0, "step": 0.05,
                    "tooltip": "细节保护强度。值越高越能保护细节，建议0.9以上"}),
                "noise_threshold": ("FLOAT", {"default": 0.02, "min": 0.005, "max": 0.1, "step": 0.005,
                    "tooltip": "噪声检测阈值。只处理变化小于此值的区域，避免破坏真实细节"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normals",)
    FUNCTION = "normal_map"

    CATEGORY = "image/filters"

    def gentle_denoise_depth(self, depth, method, strength, detail_preservation, noise_threshold):
        """温和的深度图去噪处理"""
        if method == "none":
            return depth
            
        # 只处理深度通道（第3个通道）
        depth_channel = depth[..., 2]  # 提取深度通道 [B,H,W]
        
        if method == "gentle_blur":
            # 使用PyTorch实现轻度高斯模糊
            kernel_size = max(3, int(strength * 10) | 1)  # 确保奇数
            sigma = strength * 0.5
            
            # 创建高斯核
            k = kernel_size // 2
            ax = torch.arange(-k, k + 1, dtype=torch.float32, device=depth.device)
            xx, yy = torch.meshgrid(ax, ax, indexing='ij')
            kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            kernel = kernel / kernel.sum()
            kernel = kernel.view(1, 1, kernel_size, kernel_size)
            
            # 应用高斯模糊
            depth_blurred = F.conv2d(depth_channel.unsqueeze(1), kernel, padding=k)
            depth_blurred = depth_blurred.squeeze(1)
            
            # 根据detail_preservation混合原始和模糊版本
            alpha = 1 - detail_preservation
            denoised_tensor = depth_channel * (1 - alpha) + depth_blurred * alpha
            
        else:
            # 对于cv2方法，使用极其温和的参数
            depth_np = depth_channel.cpu().numpy()
            batch_size = depth_np.shape[0]
            
            denoised_batch = []
            for i in range(batch_size):
                depth_single = depth_np[i]
                # 转换为uint8用于cv2处理
                depth_uint8 = (depth_single * 255).astype(np.uint8)
                
                if method == "nlmeans_light":
                    # 极其温和的非局部均值参数
                    h = max(0.5, strength * 2)  # 最小0.5，最大4
                    template_window = 3  # 固定最小窗口
                    search_window = 7    # 固定最小搜索窗口
                    
                    denoised = cv2.fastNlMeansDenoising(depth_uint8, 
                                                       None,
                                                       h=h,
                                                       templateWindowSize=template_window,
                                                       searchWindowSize=search_window)
                    
                elif method == "bilateral_light":
                    # 极其温和的双边滤波参数
                    d = 3  # 固定最小直径
                    sigma_color = max(5, strength * 15)  # 最小5，最大30
                    sigma_space = max(5, strength * 15)
                    
                    denoised = cv2.bilateralFilter(depth_uint8, d, sigma_color, sigma_space)
                    
                # 转回float并归一化
                denoised_float = denoised.astype(np.float32) / 255.0
                
                # 根据detail_preservation控制去噪强度
                original = depth_single
                alpha = 1 - detail_preservation
                final_denoised = original * (1 - alpha) + denoised_float * alpha
                
                denoised_batch.append(final_denoised)
            
            # 转回torch tensor
            denoised_tensor = torch.from_numpy(np.stack(denoised_batch)).to(depth.device)
        
        # 智能噪声检测：只在变化小的区域应用去噪
        if noise_threshold > 0:
            # 计算局部梯度幅度
            grad_x = torch.diff(depth_channel, dim=2, prepend=depth_channel[:, :, :1])
            grad_y = torch.diff(depth_channel, dim=1, prepend=depth_channel[:, :1, :])
            grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
            
            # 创建噪声掩码：只有梯度小的区域被认为是噪声
            noise_mask = (grad_magnitude < noise_threshold).float()
            
            # 平滑掩码边界
            noise_mask = F.conv2d(noise_mask.unsqueeze(1), 
                                torch.ones(1, 1, 3, 3, device=depth.device) / 9, 
                                padding=1).squeeze(1)
            
            # 根据掩码混合原始和去噪版本
            denoised_tensor = depth_channel * (1 - noise_mask) + denoised_tensor * noise_mask
        
        # 替换深度通道
        result = depth.clone()
        result[..., 2] = denoised_tensor
        
        return result

    def normal_map(self, depth, scale, output_mode, denoise_method="gentle_blur", 
                   denoise_strength=0.3, detail_preservation=0.95, noise_threshold=0.02):
        
        device = depth.device
        
        # 标准3x3 Sobel核（清晰无模糊）
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=device)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=device)
        kernel_x = kernel_x.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        kernel_y = kernel_y.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        
        conv2d = F.conv2d
        pad = F.pad
        
        size_x = depth.size(2)
        size_y = depth.size(1)
        max_dim = max(size_x, size_y)
        
        # 构建3D位置图
        position_map = depth.detach().clone() * scale
        xs = torch.linspace(-1 * size_x / max_dim, 1 * size_x / max_dim, steps=size_x, device=device)
        ys = torch.linspace(-1 * size_y / max_dim, 1 * size_y / max_dim, steps=size_y, device=device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
        position_map[..., 0] = grid_x.unsqueeze(0)
        position_map[..., 1] = grid_y.unsqueeze(0)
        
        # 🎯 温和的深度去噪
        if denoise_method != "none":
            position_map = self.gentle_denoise_depth(position_map, denoise_method, 
                                                   denoise_strength, detail_preservation, noise_threshold)
        
        position_map = position_map.movedim(-1, 1) # BCHW
        
        # 使用标准3x3 Sobel核计算梯度（保持清晰）
        grad_x = conv2d(pad(position_map, (1, 1, 1, 1), mode='replicate'), 
                       kernel_x, padding='valid', groups=3)
        grad_y = conv2d(pad(position_map, (1, 1, 1, 1), mode='replicate'), 
                       kernel_y, padding='valid', groups=3)
        
        # 计算法线
        cross_product = torch.cross(grad_x, grad_y, dim=1)
        normals = F.normalize(cross_product, dim=1)
        normals[:, 1] *= -1
        
        if output_mode != "Standard":
            normals[:, 0] *= -1
        
        if output_mode == "MiDaS":
            normals = torch.flip(normals, dims=[1,])
        
        normals = normals.movedim(1, -1) * 0.5 + 0.5 # BHWC
        
        return (normals,) 