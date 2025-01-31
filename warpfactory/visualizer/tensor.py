import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from .colormaps import ColorMaps

class TensorPlotter:
    """Visualize tensor components and properties."""
    
    def __init__(self):
        self.cmaps = ColorMaps()
    
    def plot_component(self, metric: Dict[str, np.ndarray], component: str,
                      x: np.ndarray, y: np.ndarray) -> plt.Figure:
        """Plot a single metric component.
        
        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric tensor components
        component : str
            Component to plot (e.g., 'g_tt')
        x, y : np.ndarray
            Coordinate arrays
            
        Returns
        -------
        plt.Figure
            Figure handle
        """
        X, Y = np.meshgrid(x, y)
        
        fig, ax = plt.subplots()
        im = ax.pcolormesh(X, Y, metric[component], cmap=self.cmaps.redblue(),
                          shading='auto')
        plt.colorbar(im, ax=ax)
        ax.set_title(f'Metric Component {component}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        
        return fig
    
    def plot_heatmap(self, metric: Dict[str, np.ndarray], x: np.ndarray,
                     y: np.ndarray) -> plt.Figure:
        """Plot heatmap of all metric components.
        
        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric tensor components
        x, y : np.ndarray
            Coordinate arrays
            
        Returns
        -------
        plt.Figure
            Figure handle
        """
        components = list(metric.keys())
        n = len(components)
        
        fig, axes = plt.subplots(n, n, figsize=(3*n, 3*n))
        
        for i, comp1 in enumerate(components):
            for j, comp2 in enumerate(components):
                ax = axes[i, j]
                if i <= j:  # Upper triangular part
                    im = ax.pcolormesh(metric[comp1], cmap=self.cmaps.redblue(),
                                     shading='auto')
                    plt.colorbar(im, ax=ax)
                ax.set_title(f'{comp1}-{comp2}')
                ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig
    
    def plot_eigenvalues(self, metric: Dict[str, np.ndarray], x: np.ndarray,
                        y: np.ndarray) -> plt.Figure:
        """Plot eigenvalues of the metric tensor.
        
        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric tensor components
        x, y : np.ndarray
            Coordinate arrays
            
        Returns
        -------
        plt.Figure
            Figure handle
        """
        # Convert metric to matrix form
        n = len(x)
        matrix = np.zeros((n, n, 2, 2))  # 2x2 matrix at each point
        
        matrix[:, :, 0, 0] = metric["g_tt"]
        matrix[:, :, 0, 1] = matrix[:, :, 1, 0] = metric.get("g_tx", np.zeros((n, n)))
        matrix[:, :, 1, 1] = metric.get("g_xx", np.ones((n, n)))
        
        # Calculate eigenvalues
        eigenvals = np.linalg.eigvals(matrix)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        X, Y = np.meshgrid(x, y)
        im1 = ax1.pcolormesh(X, Y, eigenvals[:, :, 0].real, cmap=self.cmaps.redblue(),
                            shading='auto')
        plt.colorbar(im1, ax=ax1)
        ax1.set_title('First Eigenvalue')
        ax1.set_aspect('equal')
        
        im2 = ax2.pcolormesh(X, Y, eigenvals[:, :, 1].real, cmap=self.cmaps.redblue(),
                            shading='auto')
        plt.colorbar(im2, ax=ax2)
        ax2.set_title('Second Eigenvalue')
        ax2.set_aspect('equal')
        
        plt.tight_layout()
        return fig