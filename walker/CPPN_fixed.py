import numpy as np

from walker.CPPN_mutable import query_cppn
from walker.walk_creator import walker_creator


def make_walker():
    wc = walker_creator()

    def connect(x1, y1, x2, y2):
        if ((x1 - x2) ** 2 + (y1 - y2) ** 2) > 4.5:
            return False
        return True

    def amp(x1, y1, x2, y2):
        return max(abs(x1 - x2), abs(y1 - y2))

    def phase(x1, y1, x2, y2):
        return np.sign(x1)

    joints = query_cppn(wc, 8, 3, 1.5, connect, amp, phase)

    return wc.get_walker()
