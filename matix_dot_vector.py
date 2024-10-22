def matrix_dot_vector(a:list[list[int|float]],b:list[int|float])-> list[int|float]:
    if len(a[0]) != len(b):
        return -1

    res = []

    for row in a:
        res.append(sum(i * j for i, j in zip(row, b)))
        
    return res

print(matrix_dot_vector([[1,2],[2,4],[6,8],[12,4]],[1,2,3]))
