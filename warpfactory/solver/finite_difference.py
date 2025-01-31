import numpy as np

class FiniteDifference:
    """Finite difference methods for numerical derivatives."""
    
    def derivative1(self, f: np.ndarray, x: np.ndarray, axis: int = 0) -> np.ndarray:
        """Calculate first derivative using central differences.
        
        Parameters
        ----------
        f : np.ndarray
            Function values
        x : np.ndarray
            Coordinate values
        axis : int, optional
            Axis along which to take derivative
            
        Returns
        -------
        np.ndarray
            First derivative df/dx
        """
        dx = x[1] - x[0]  # Assume uniform grid
        
        # Use central differences for interior points
        # For better accuracy, use explicit central difference formula
        df = np.zeros_like(f)
        
        # Interior points
        df[1:-1] = (f[2:] - f[:-2]) / (2 * dx)
        
        # Forward difference at left boundary
        df[0] = (f[1] - f[0]) / dx
        
        # Backward difference at right boundary
        df[-1] = (f[-1] - f[-2]) / dx
        
        return df
    
    def derivative2(self, f: np.ndarray, x: np.ndarray, axis: int = 0) -> np.ndarray:
        """Calculate second derivative using central differences.
        
        Parameters
        ----------
        f : np.ndarray
            Function values
        x : np.ndarray
            Coordinate values
        axis : int, optional
            Axis along which to take derivative
            
        Returns
        -------
        np.ndarray
            Second derivative d²f/dx²
        """
        dx = x[1] - x[0]  # Assume uniform grid
        
        # Initialize output array
        d2f = np.zeros_like(f)
        
        # Use central differences for interior points
        # d²f/dx² ≈ (f[i+1] - 2f[i] + f[i-1])/dx²
        d2f[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / dx**2
        
        # Forward difference for left boundary
        d2f[0] = (f[2] - 2*f[1] + f[0]) / dx**2
        
        # Backward difference for right boundary
        d2f[-1] = (f[-1] - 2*f[-2] + f[-3]) / dx**2
        
        return d2f