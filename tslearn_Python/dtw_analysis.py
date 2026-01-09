import numpy as np
from tslearn.metrics import dtw


def compute_dtw_distance(ts1, ts2):
    return dtw(ts1, ts2)


if __name__ == "__main__":
    ts1 = np.array([1, 2, 3, 4, 5])
    ts2 = np.array([2, 3, 4, 5, 6])
    print(f"DTW Distance: {compute_dtw_distance(ts1, ts2):.2f}")
