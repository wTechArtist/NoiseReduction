from .NoiseReduction import *
from .DepthToNormals import *

# 注册节点映射
NODE_CLASS_MAPPINGS = {
    "NoiseReduction": NoiseReduction,
    "DepthToNormals": DepthToNormals
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NoiseReduction": "Reduce Noise Filter",
    "DepthToNormals": "VVL Depth to Normals"
}
