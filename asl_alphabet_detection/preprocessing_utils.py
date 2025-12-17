"""
Preprocessing utilities for ASL hand gesture recognition.
This module must be imported in inference code to ensure consistent preprocessing.
"""
import numpy as np
import math

NUM_HANDS = 2
NUM_LANDMARKS = 21
COORDS = 3
VEC_LEN = NUM_HANDS * NUM_LANDMARKS * COORDS
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9


def reshape_landmarks(vec):
    return vec.reshape(NUM_HANDS, NUM_LANDMARKS, COORDS)


def normalize_per_hand(X_arr: np.ndarray) -> np.ndarray:
    """Normalize landmarks per hand: translate by wrist, scale by wrist->middle_mcp distance."""
    Xn = X_arr.copy()
    if Xn.ndim == 1:
        Xn = Xn.reshape(1, -1)
    Xn = Xn.reshape(-1, NUM_HANDS, NUM_LANDMARKS, COORDS)
    for i in range(Xn.shape[0]):
        for h in range(NUM_HANDS):
            hand = Xn[i, h]
            if np.allclose(hand, 0.0):
                continue
            wrist = hand[WRIST_IDX]
            hand[:, :2] -= wrist[:2]  # translate by wrist in x,y; keep z as-is
            # scale by distance wrist->middle_mcp on xy plane
            ref = hand[MIDDLE_MCP_IDX]
            scale = np.linalg.norm(ref[:2])
            if scale > 1e-6:
                hand[:, :2] /= scale
            Xn[i, h] = hand
    return Xn.reshape(-1, VEC_LEN)

