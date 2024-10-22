import numpy as np

def compute_qkv(X, W_q, W_k, W_v) -> tuple[np.array, np.array, np.array]:
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    return Q, K, V



def self_attention(Q, K, V) -> float:
    scores = Q @ K.T
    d_k = K.shape[1]
    scores = scores / np.sqrt(d_k)
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=1, keepdims=True)
    attention = weights @ V

    return attention



X = np.array([[1, 0], [0, 1]])
W_q = np.array([[1, 0], [0, 1]])
W_k = np.array([[1, 0], [0, 1]])
W_v = np.array([[1, 2], [3, 4]])

Q, K, V = compute_qkv(X, W_q, W_k, W_v)
output = self_attention(Q, K, V)

print(output)

# Expected Output:
# [[1.660477 2.660477]
#  [2.339523 3.339523]]

# test 3

X = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
W_q = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
W_k = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
W_v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

Q, K, V = compute_qkv(X, W_q, W_k, W_v)
output = self_attention(Q, K, V)
print(output)

# [[8.0, 10.0, 12.0], [8.61987385, 10.61987385, 12.61987385], [7.38012615, 9.38012615, 11.38012615]]