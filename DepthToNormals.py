import torch
import torch.nn.functional as F


class DepthToNormals:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "depth": ("IMAGE",),
                "scale": ("FLOAT",{"default": 1, "min": 0.001, "max": 1000, "step": 0.001}),
                "output_mode": (["Standard", "BAE", "MiDaS"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normals",)
    FUNCTION = "normal_map"

    CATEGORY = "image/filters"

    def normal_map(self, depth, scale, output_mode):
        kernel_x = torch.Tensor([[0,0,0],[1,0,-1],[0,0,0]]).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        kernel_y = torch.Tensor([[0,1,0],[0,0,0],[0,-1,0]]).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        conv2d = F.conv2d
        pad = F.pad
        
        size_x = depth.size(2)
        size_y = depth.size(1)
        max_dim = max(size_x, size_y)
        position_map = depth.detach().clone() * scale
        xs = torch.linspace(-1 * size_x / max_dim, 1 * size_x / max_dim, steps=size_x)
        ys = torch.linspace(-1 * size_y / max_dim, 1 * size_y / max_dim, steps=size_y)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
        position_map[..., 0] = grid_x.unsqueeze(0)
        position_map[..., 1] = grid_y.unsqueeze(0)
        
        position_map = position_map.movedim(-1, 1) # BCHW
        grad_x = conv2d(pad(position_map, (1,1,1,1), mode='replicate'), kernel_x, padding='valid', groups=3)
        grad_y = conv2d(pad(position_map, (1,1,1,1), mode='replicate'), kernel_y, padding='valid', groups=3)
        
        cross_product = torch.cross(grad_x, grad_y, dim=1)
        normals = F.normalize(cross_product)
        normals[:, 1] *= -1
        
        if output_mode != "Standard":
            normals[:, 0] *= -1
        
        if output_mode == "MiDaS":
            normals = torch.flip(normals, dims=[1,])
        
        normals = normals.movedim(1, -1) * 0.5 + 0.5 # BHWC
        return (normals,) 