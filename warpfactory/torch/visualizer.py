"""PyTorch-based tensor field visualization."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Union

class TorchFieldVisualizer:
    """GPU-accelerated field visualization."""
    
    def __init__(self, device: Union[str, torch.device] = "cuda"):
        """Initialize visualizer.
        
        Parameters
        ----------
        device : str or torch.device
            Device to use for computations
        """
        self.device = torch.device(device)
    
    def plot_scalar_field(self, field: torch.Tensor, x: torch.Tensor,
                         y: torch.Tensor) -> plt.Figure:
        """Plot scalar field.
        
        Parameters
        ----------
        field : torch.Tensor
            Scalar field data
        x, y : torch.Tensor
            Coordinate arrays
            
        Returns
        -------
        plt.Figure
            Figure handle
        """
        # Move data to CPU for plotting
        field_np = field.cpu().numpy()
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        
        fig, ax = plt.subplots()
        X, Y = np.meshgrid(x_np, y_np)
        im = ax.pcolormesh(X, Y, field_np, shading='auto')
        plt.colorbar(im, ax=ax)
        ax.set_aspect('equal')
        
        return fig
    
    def plot_vector_field(self, vectors: torch.Tensor, x: torch.Tensor,
                         y: torch.Tensor) -> plt.Figure:
        """Plot vector field.
        
        Parameters
        ----------
        vectors : torch.Tensor
            Vector field data [3, nx, ny]
        x, y : torch.Tensor
            Coordinate arrays
            
        Returns
        -------
        plt.Figure
            Figure handle
        """
        # Move data to CPU for plotting
        vx = vectors[0].cpu().numpy()
        vy = vectors[1].cpu().numpy()
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        
        fig, ax = plt.subplots()
        X, Y = np.meshgrid(x_np, y_np)
        
        # Downsample for clarity
        skip = 4
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                 vx[::skip, ::skip], vy[::skip, ::skip])
        ax.set_aspect('equal')
        
        return fig
    
    def plot_streamlines(self, vectors: torch.Tensor, x: torch.Tensor,
                        y: torch.Tensor) -> plt.Figure:
        """Plot streamlines.
        
        Parameters
        ----------
        vectors : torch.Tensor
            Vector field data [3, nx, ny]
        x, y : torch.Tensor
            Coordinate arrays
            
        Returns
        -------
        plt.Figure
            Figure handle
        """
        # Move data to CPU for plotting
        vx = vectors[0].cpu().numpy()
        vy = vectors[1].cpu().numpy()
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        
        fig, ax = plt.subplots()
        X, Y = np.meshgrid(x_np, y_np)
        
        ax.streamplot(X, Y, vx, vy)
        ax.set_aspect('equal')
        
        return fig
    
    def animate_field_evolution(self, field: torch.Tensor,
                              vectors: torch.Tensor,
                              x: torch.Tensor, y: torch.Tensor,
                              frames: int = 30) -> FuncAnimation:
        """Create animation of field evolution.
        
        Parameters
        ----------
        field : torch.Tensor
            Initial scalar field
        vectors : torch.Tensor
            Vector field for evolution
        x, y : torch.Tensor
            Coordinate arrays
        frames : int
            Number of animation frames
            
        Returns
        -------
        FuncAnimation
            Animation object
        """
        fig, ax = plt.subplots()
        X, Y = np.meshgrid(x.cpu().numpy(), y.cpu().numpy())
        
        # Initialize plot
        field_np = field.cpu().numpy()
        im = ax.pcolormesh(X, Y, field_np, shading='auto')
        plt.colorbar(im, ax=ax)
        ax.set_aspect('equal')
        
        # Animation update function
        def update(frame):
            # Evolve field using vectors (simplified)
            dt = 0.1
            vx = vectors[0].cpu().numpy()
            vy = vectors[1].cpu().numpy()
            
            # Simple advection
            field_np = field.cpu().numpy()
            field_next = field_np - dt * (
                vx * np.gradient(field_np, axis=1) +
                vy * np.gradient(field_np, axis=0)
            )
            
            im.set_array(field_next.ravel())
            return [im]
        
        anim = FuncAnimation(fig, update, frames=frames,
                           interval=50, blit=True)
        return anim