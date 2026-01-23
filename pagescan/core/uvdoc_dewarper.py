"""
UVDoc-based deep learning dewarper wrapper.

Integrates the UVDoc model (SIGGRAPH 2023) for state-of-the-art document dewarping.
"""

import sys
import os
import numpy as np
import cv2
from typing import Optional
from pathlib import Path

# Add UVDoc to path
UVDOC_PATH = Path(__file__).parent.parent.parent / "external" / "UVDoc"
if str(UVDOC_PATH) not in sys.path:
    sys.path.insert(0, str(UVDOC_PATH))


class UVDocDewarper:
    """
    Deep learning based dewarper using UVDoc (SIGGRAPH 2023).
    
    This provides state-of-the-art document dewarping using a neural 
    grid-based approach with pretrained weights.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the UVDoc dewarper.
        
        Args:
            model_path: Path to model weights. If None, uses default.
        """
        self._model = None
        self._device = None
        
        if model_path is None:
            model_path = str(UVDOC_PATH / "model" / "best_model.pkl")
        
        self.model_path = model_path
        self._loaded = False
        
    def _load_model(self):
        """Lazy load the model."""
        if self._loaded:
            return
            
        try:
            import torch
            from model import UVDocnet
            from utils import IMG_SIZE
            
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load model with CPU fallback for CUDA-trained weights
            model = UVDocnet(num_filter=32, kernel_size=5)
            ckpt = torch.load(self.model_path, map_location=self._device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            
            self._model = model
            self._model.to(self._device)
            self._model.eval()
            self._img_size = IMG_SIZE
            self._loaded = True
            print(f"UVDoc model loaded on {self._device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load UVDoc model: {e}")
    
    def dewarp(self, image: np.ndarray) -> np.ndarray:
        """
        Dewarp a document image using UVDoc.
        
        Args:
            image: Input image as RGB numpy array (H, W, 3)
            
        Returns:
            Dewarped image as RGB numpy array
        """
        self._load_model()
        
        import torch
        from utils import bilinear_unwarping
        
        # Prepare input
        img = image.astype(np.float32) / 255.0
        inp = torch.from_numpy(
            cv2.resize(img, tuple(self._img_size)).transpose(2, 0, 1)
        ).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            inp = inp.to(self._device)
            point_positions2D, _ = self._model(inp)
        
        # Unwarp at original resolution
        size = image.shape[:2][::-1]  # (W, H)
        warped_tensor = torch.from_numpy(
            img.transpose(2, 0, 1)
        ).unsqueeze(0).to(self._device)
        
        unwarped = bilinear_unwarping(
            warped_img=warped_tensor,
            point_positions=torch.unsqueeze(point_positions2D[0], dim=0),
            img_size=tuple(size),
        )
        
        # Convert back to numpy
        result = (unwarped[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        
        return result
    
    def is_available(self) -> bool:
        """Check if PyTorch and model are available."""
        try:
            import torch
            return os.path.exists(self.model_path)
        except ImportError:
            return False


def test_uvdoc():
    """Quick test of UVDoc dewarper."""
    dewarper = UVDocDewarper()
    
    if not dewarper.is_available():
        print("UVDoc not available (missing torch or model)")
        return
    
    # Create test image
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    result = dewarper.dewarp(test_img)
    print(f"Input shape: {test_img.shape}, Output shape: {result.shape}")
    print("UVDoc test passed!")


if __name__ == "__main__":
    test_uvdoc()
