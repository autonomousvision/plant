import math


def get_coords(self, x, y, angle, extent_x, extent_y):
    endx1 = x - extent_x * math.sin(math.radians(angle)) - extent_y * math.cos(math.radians(angle))
    endy1 = y + extent_x * math.cos(math.radians(angle)) - extent_y * math.sin(math.radians(angle))

    endx2 = x + extent_x * math.sin(math.radians(angle)) - extent_y * math.cos(math.radians(angle))
    endy2 = y - extent_x * math.cos(math.radians(angle)) - extent_y * math.sin(math.radians(angle))

    endx3 = x + extent_x * math.sin(math.radians(angle)) + extent_y * math.cos(math.radians(angle))
    endy3 = y - extent_x * math.cos(math.radians(angle)) + extent_y * math.sin(math.radians(angle))

    endx4 = x - extent_x * math.sin(math.radians(angle)) + extent_y * math.cos(math.radians(angle))
    endy4 = y + extent_x * math.cos(math.radians(angle)) + extent_y * math.sin(math.radians(angle))

    return (endx1, endy1), (endx2, endy2), (endx3, endy3), (endx4, endy4)