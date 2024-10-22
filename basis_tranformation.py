import numpy as np

def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
	P = np.linalg.inv(C) @ B
	return P

print(transform_basis([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 2.3, 3], [4.4, 25, 6], [7.4, 8, 9]]))
