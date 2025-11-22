"""
Custom ComfyUI Nodes Package
Organized structure for better maintainability
"""

# Import Zy nodes (already consolidated)
from .zy_nodes import NODE_CLASS_MAPPINGS as ZY_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as ZY_DISPLAY

# Import My nodes
from .my_nodes import NODE_CLASS_MAPPINGS as MY_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MY_DISPLAY

# Import My mask nodes
from .my_mask_nodes import NODE_CLASS_MAPPINGS as MY_MASK_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MY_MASK_DISPLAY

# Combine all mappings
NODE_CLASS_MAPPINGS = {**ZY_MAPPINGS, **MY_MAPPINGS, **MY_MASK_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**ZY_DISPLAY, **MY_DISPLAY, **MY_MASK_DISPLAY}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
