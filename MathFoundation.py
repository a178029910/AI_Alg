import torch

# 对称矩阵的特征分解
A = torch.tensor([[4, 1, 2], [1, 3, 0], [2, 0, 5]], dtype=torch.float64)
eigenvalues, eigenvectors = torch.linalg.eig(A)
print("特征值:", eigenvalues.real)
print("特征向量:\n", eigenvectors.real)

# SVD 分解
M = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float64)
U, S, Vt = torch.linalg.svd(M)
print("左奇异向量 U:\n", U)
print("奇异值 S:", S)
print("右奇异向量 V^T:\n", Vt.T)

# PCA 降维：保留前 k 个奇异值
k = 1
M_reduced = U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]
print(f"PCA 降维 (k={k}):\n", M_reduced)