import numpy as np

# Define domains
K = 2  # Each variable can take values 0 or 1

# Define factors
# p(c)
f3 = np.array([0.6, 0.4])  # shape: (c,)
# p(d)
f4 = np.array([0.7, 0.3])  # shape: (d,)
# p(e|d)
f5 = np.array([[0.9, 0.2],   # p(e=0|d=0), p(e=0|d=1)
               [0.1, 0.8]])  # p(e=1|d=0), p(e=1|d=1)
# p(b|c,d)
f2 = np.random.rand(K, K, K)  # shape: (b, c, d)
f2 /= f2.sum(axis=0, keepdims=True)  # Normalize over b
# p(a|b)
f1 = np.array([[0.3, 0.9],   # p(a=0|b=0), p(a=0|b=1)
               [0.7, 0.1]])  # p(a=1|b=0), p(a=1|b=1)

# Messages
# c → f2
msg_c_to_f2 = f3.copy()  # shape: (c,)
# d → f2
msg_d_to_f2 = f4.copy()  # shape: (d,)
# d → f5 = f4
msg_d_to_f5 = f4.copy()  # same as f4
# f5 → d
msg_f5_to_d = np.sum(f5, axis=0)  # sum over e, shape: (d,)
# Updated d → f2
msg_d_to_f2 = msg_d_to_f2 * msg_f5_to_d  # combine both messages

# f2 → b
msg_f2_to_b = np.zeros(K)
for b in range(K):
    for c in range(K):
        for d in range(K):
            msg_f2_to_b[b] += f2[b, c, d] * msg_c_to_f2[c] * msg_d_to_f2[d]

# b → f1
msg_b_to_f1 = msg_f2_to_b.copy()  # shape: (b,)
# f1 → a
msg_f1_to_a = np.zeros(K)
for a in range(K):
    for b in range(K):
        msg_f1_to_a[a] += f1[a, b] * msg_b_to_f1[b]

# Normalize final marginal
p_a = msg_f1_to_a / msg_f1_to_a.sum()

print("Marginal p(a):", p_a)

