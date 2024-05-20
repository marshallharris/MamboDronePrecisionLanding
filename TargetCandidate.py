from dataclasses import dataclass
import math
import numpy as np

@dataclass
class TargetCandidate:
    points: np.ndarray
    avgBrightness: float
    cornerAngles: np.ndarray
    area: float


    def containsRightAngles(self):
        for angle in self.cornerAngles:
            if math.isclose(angle, math.pi/2, abs_tol=math.pi/18):
                return True
        return False