import numpy as np

def build_windows_old(data_120):
    T, D = data_120.shape
    X_out = []
    for t in range(T):
        row = []
        for j in range(6):
            idx = t - j
            if idx < 0:
                row.extend([0]*D)
            else:
                row.extend(data_120[idx])
        X_out.append(row)
    return np.array(X_out)

def build_windows_new(data_120):
    T, D = data_120.shape
    padded = np.vstack([np.zeros((5, D), dtype=np.float32), data_120.astype(np.float32)])
    X_out = np.zeros((T, 6 * D), dtype=np.float32)
    for j in range(6):
        X_out[:, j*D : (j+1)*D] = padded[5-j : 5-j+T]
    return X_out

data = np.arange(10).reshape(5, 2)
print("Data:\n", data)
old = build_windows_old(data)
new = build_windows_new(data)
print("Old equals new:", np.allclose(old, new))
