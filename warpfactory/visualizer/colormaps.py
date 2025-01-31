import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class ColorMaps:
    """Custom colormaps for visualization."""
    
    def redblue(self, N: int = 256) -> LinearSegmentedColormap:
        """Create a diverging red-white-blue colormap.
        
        Parameters
        ----------
        N : int, optional
            Number of colors
            
        Returns
        -------
        LinearSegmentedColormap
            Matplotlib colormap
        """
        # Define colors
        colors = [
            (0.0, (0.23137255, 0.298039216, 0.752941176)),  # Dark blue
            (0.5, (1.0, 1.0, 1.0)),  # White
            (1.0, (0.705882353, 0.015686275, 0.149019608))  # Dark red
        ]
        
        return LinearSegmentedColormap.from_list("redblue", colors, N=N)
    
    def warp(self, N: int = 256) -> LinearSegmentedColormap:
        """Create a custom colormap for warp drive visualization.
        
        Parameters
        ----------
        N : int, optional
            Number of colors
            
        Returns
        -------
        LinearSegmentedColormap
            Matplotlib colormap
        """
        # Define colors
        colors = [
            (0.0, (0.0, 0.0, 0.2)),  # Dark blue
            (0.3, (0.0, 0.0, 1.0)),  # Blue
            (0.5, (0.0, 1.0, 1.0)),  # Cyan
            (0.7, (1.0, 1.0, 0.0)),  # Yellow
            (1.0, (1.0, 0.0, 0.0))   # Red
        ]
        
        return LinearSegmentedColormap.from_list("warp", colors, N=N)