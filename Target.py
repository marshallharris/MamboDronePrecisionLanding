from dataclasses import dataclass

@dataclass
class Target:
    x_center: float
    y_center: float
    width: float
    height: float
    rho: float
    phi: float
    timestamp: float