import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from .colormaps import ColorMaps

class ThreePlusOnePlotter:
    """Visualize 3+1 decomposition components."""
    
    def __init__(self):
        self.cmaps = ColorMaps()
    
    def plot_lapse(self, decomp: Dict, x: np.ndarray, y: np.ndarray) -> plt.Figure:
        """Plot lapse function.
        
        Parameters
        ----------
        decomp : Dict
            3+1 decomposition data
        x, y : np.ndarray
            Coordinate arrays
            
        Returns
        -------
        plt.Figure
            Figure handle
        """
        X, Y = np.meshgrid(x, y)
        
        fig, ax = plt.subplots()
        im = ax.pcolormesh(X, Y, decomp["alpha"], cmap=self.cmaps.warp(),
                          shading='auto')
        plt.colorbar(im, ax=ax)
        ax.set_title('Lapse Function α')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        
        return fig
    
    def plot_shift(self, decomp: Dict, x: np.ndarray, y: np.ndarray) -> plt.Figure:
        """Plot shift vector field.
        
        Parameters
        ----------
        decomp : Dict
            3+1 decomposition data
        x, y : np.ndarray
            Coordinate arrays
            
        Returns
        -------
        plt.Figure
            Figure handle
        """
        X, Y = np.meshgrid(x, y)
        
        fig, ax = plt.subplots()
        
        # Plot shift vector magnitude
        beta_mag = np.sqrt(decomp["beta"]["x"]**2 + decomp["beta"]["y"]**2)
        im = ax.pcolormesh(X, Y, beta_mag, cmap=self.cmaps.warp(),
                          shading='auto')
        plt.colorbar(im, ax=ax)
        
        # Plot shift vector field
        skip = 4  # Plot every nth vector
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                 decomp["beta"]["x"][::skip, ::skip],
                 decomp["beta"]["y"][::skip, ::skip],
                 color='white', alpha=0.5)
        
        ax.set_title('Shift Vector β')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        
        return fig
    
    def plot_spatial_metric(self, decomp: Dict, x: np.ndarray,
                          y: np.ndarray) -> plt.Figure:
        """Plot spatial metric components.
        
        Parameters
        ----------
        decomp : Dict
            3+1 decomposition data
        x, y : np.ndarray
            Coordinate arrays
            
        Returns
        -------
        plt.Figure
            Figure handle
        """
        X, Y = np.meshgrid(x, y)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot γ_xx
        im1 = ax1.pcolormesh(X, Y, decomp["gamma"]["xx"], cmap=self.cmaps.redblue(),
                            shading='auto')
        plt.colorbar(im1, ax=ax1)
        ax1.set_title('γ_xx')
        ax1.set_aspect('equal')
        
        # Plot γ_xy
        im2 = ax2.pcolormesh(X, Y, decomp["gamma"]["xy"], cmap=self.cmaps.redblue(),
                            shading='auto')
        plt.colorbar(im2, ax=ax2)
        ax2.set_title('γ_xy')
        ax2.set_aspect('equal')
        
        # Plot γ_yy
        im3 = ax3.pcolormesh(X, Y, decomp["gamma"]["yy"], cmap=self.cmaps.redblue(),
                            shading='auto')
        plt.colorbar(im3, ax=ax3)
        ax3.set_title('γ_yy')
        ax3.set_aspect('equal')
        
        plt.tight_layout()
        return fig