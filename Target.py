from dataclasses import dataclass

@dataclass
class Target:
    x_center: float
    y_center: float
    area: float
    width: float
    height: float
    rho: float
    phi: float
    timestamp: float

    def stringForImage(self):
        return f"x: {self.x_center}, y: {self.y_center}, w: {self.width}, h: {self.height}, rho:{self.rho}"

    def percentOfImage(self):
        imageArea = 640 * 480
        return self.area * 100 / imageArea