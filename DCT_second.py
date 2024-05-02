def DCT_process(matrix, i, j):
    width = matrix.shape[1]
    height = matrix.shape[0]
    value = 0.
    for col in range(height):
        for row in range(width):
            save = matrix[col, row]
            save *= math.cos(math.pi * (2 * col + 1) * i / (2. * height))
            save *= math.cos(math.pi * (2 * row + 1) * j / (2. * width))
            value += save
    c = 1.
    if i == 0:
        c /= np.sqrt(2)
    if j == 0:
        c /= np.sqrt(2)

    return (2. / np.sqrt(height * width)) * c * value


def DCT(matrix):
    width = matrix.shape[1]
    height = matrix.shape[0]
    dct = np.zeros_like(matrix)

    for col in range(height):
        for row in range(width):
            dct[col, row] = DCT_process(matrix, col, row)
    return dct


def IDCT_process(dct, i, j):
    width = dct.shape[1]
    height = dct.shape[0]
    value = 0

    for col in range(height):
        for row in range(width):
            save = dct[col, row]
            if col == 0:
                save /= np.sqrt(2)
            if row == 0:
                save /= np.sqrt(2)
            save *= math.cos(math.pi * (2 * i + 1) * col / (2. * height))
            save *= math.cos(math.pi * (2 * j + 1) * row / (2. * width))
            value += save

    return (2. / np.sqrt(height * width)) * value


def IDCT(dct):
    width = dct.shape[1]
    height = dct.shape[0]
    matrix = np.zeros_like(dct)

    for col in range(height):
        for row in range(width):
            matrix[col, row] = IDCT_process(dct, col, row)
    return matrix


value = np.arange(10, 160, 10, dtype=np.float)
matrix = np.reshape(value, (5, 3))
matrix_dct = DCT(matrix)
print(matrix_dct.astype(np.int))
print(IDCT(matrix_dct))
