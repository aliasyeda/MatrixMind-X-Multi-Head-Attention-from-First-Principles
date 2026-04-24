import numpy as np

# -----------------------------
# Softmax
# -----------------------------
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# -----------------------------
# Single Attention Head
# -----------------------------
def attention(Q, K, V):
    scores = np.dot(Q, K.T)
    d_k = K.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)
    weights = softmax(scaled_scores)
    output = np.dot(weights, V)
    return output, weights

# -----------------------------
# Multi-Head Attention
# -----------------------------
def multi_head_attention(X, num_heads=2):
    d_model = X.shape[1]
    head_dim = d_model // num_heads

    heads_output = []

    for i in range(num_heads):
        # Split input for each head
        start = i * head_dim
        end = start + head_dim

        Q = X[:, start:end]
        K = X[:, start:end]
        V = X[:, start:end]

        out, _ = attention(Q, K, V)
        heads_output.append(out)

    # Concatenate outputs
    final_output = np.concatenate(heads_output, axis=-1)

    return final_output

# -----------------------------
# Example Input
# -----------------------------
X = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 1, 0, 0]
])

result = multi_head_attention(X, num_heads=2)

print("Final Multi-Head Attention Output:")
print(result)