from enum import IntEnum
import numpy as np

class Command(IntEnum):
    ATTACK = 0
    DEFEND = 1
    HOLD = 2
    FLANK_L = 3
    FLANK_R = 4
    RETREAT = 5

N_COMMANDS = 6

def onehot_cmd(cmd: int, n: int = N_COMMANDS):
    v = np.zeros((n,), dtype=np.float32)
    v[int(cmd)] = 1.0
    return v
