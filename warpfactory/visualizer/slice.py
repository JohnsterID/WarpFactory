import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from .colormaps import ColorMaps

class SliceData:
    """Extract and visualize slices of 3D data."""
    
    def __init__(self):
        self.cmaps = ColorMaps()
    
    def get_slice(self, data: np.ndarray, x: np.ndarray, y: np.ndarray,
                 z: np.ndarray, plane: str = 'xy', coord: float = 0.0) -> np.ndarray:
        """Extract a 2D slice from 3D data.
        
        Parameters
        ----------
        data : np.ndarray
            3D data array
        x, y, z : np.ndarray
            Coordinate arrays
        plane : str
            Plane to slice ('xy', 'xz', or 'yz')
        coord : float
            Coordinate value along perpendicular direction
            
        Returns
        -------
        np.ndarray
            2D slice of data
        """
        if plane == 'xy':
            idx = np.argmin(np.abs(z - coord))
            return data[:, :, idx]
        elif plane == 'xz':
            idx = np.argmin(np.abs(y - coord))
            return data[:, idx, :]
        elif plane == 'yz':
            idx = np.argmin(np.abs(x - coord))
            return data[idx, :, :]
        else:
            raise ValueError(f"Unknown plane: {plane}")
    
    def plot_slice(self, slice_data: np.ndarray, x: np.ndarray, y: np.ndarray,
                  title: str = '', xlabel: str = 'x', ylabel: str = 'y',
                  cmap: str = None) -> plt.Figure:
        """Plot a 2D slice.
        
        Parameters
        ----------
        slice_data : np.ndarray
            2D slice data
        x, y : np.ndarray
            Coordinate arrays
        title : str, optional
            Plot title
        xlabel : str, optional
            Label for x-axis
        ylabel : str, optional
            Label for y-axis
        cmap : str, optional
            Name of colormap to use (if None, uses default warp colormap)
            
        Returns
        -------
        plt.Figure
            Figure handle
        """
        X, Y = np.meshgrid(x, y)
        
        fig, ax = plt.subplots()
        im = ax.pcolormesh(X, Y, slice_data, 
                          cmap=plt.get_cmap(cmap) if cmap else self.cmaps.warp(),
                          shading='auto')
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal')
        
        return fig