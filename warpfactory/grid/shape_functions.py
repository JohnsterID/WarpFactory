"""Shape functions shared by the grid metric builders.

Ports of MATLAB shapeFunction_Alcubierre.m and compactSigmoid.m in
vectorized form.
"""

import numpy as np


def alcubierre_shape(r: np.ndarray, R: float, sigma: float) -> np.ndarray:
    """Alcubierre top-hat: f -> 1 inside radius R, f -> 0 far outside.

    f(r) = [tanh(sigma(R + r)) + tanh(sigma(R - r))] / [2 tanh(sigma R)]
    """
    return (np.tanh(sigma*(R + r)) + np.tanh(sigma*(R - r))) / (2*np.tanh(sigma*R))


def compact_sigmoid(r: np.ndarray, R1: float, R2: float, sigma: float,
                    Rbuff: float = 0.0) -> np.ndarray:
    """Compactly supported sigmoid falling from 1 at R1+Rbuff to 0 at R2-Rbuff.

    Port of compactSigmoid.m: exactly 1 for r <= R1+Rbuff, exactly 0 for
    r >= R2-Rbuff, with a smooth monotone transition across the shell
    wall. Used as the interior shift profile of the warp shell.
    """
    r = np.asarray(r, dtype=float)
    inner = R1 + Rbuff
    outer = R2 - Rbuff
    interior = (r > inner) & (r < outer)

    f = np.where(r <= inner, 1.0, 0.0)
    r_in = r[interior]
    # exponent -> +inf at the inner edge and -inf at the outer edge;
    # clamp to avoid exp overflow (result saturates at 0/1 anyway)
    exponent = np.clip(
        (R2 - R1 - 2*Rbuff)*(sigma + 2)/2
        * (1.0/(r_in - outer) + 1.0/(r_in - inner)), -700, 700)
    f[interior] = 1.0 - 1.0/(np.exp(exponent) + 1.0)

    if np.any(~np.isfinite(f)):
        raise ValueError("compact_sigmoid produced non-finite values")
    return f
