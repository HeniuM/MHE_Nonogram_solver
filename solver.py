from utils import generate_possible_sequences, is_valid_sequence


def solve_nonogram(grid, row_clues, col_clues, row_sequences, col_sequences, row=0, col=0):
    """
    Rozwiązuje nonogram przy użyciu algorytmu przeszukiwania z powrotem (backtracking).
    """
    if row == len(grid):
        return grid

    next_row, next_col = (row, col + 1) if col + 1 < len(grid[0]) else (row + 1, 0)

    for val in [1, 0]:
        grid[row][col] = val
        print(f"Próba wartości {val} na pozycji ({row}, {col})")

        row_valid = any(is_valid_sequence(grid, seq, row, True) for seq in row_sequences[row])
        col_valid = any(is_valid_sequence(grid, seq, col, False) for seq in col_sequences[col])

        if row_valid and col_valid:
            result = solve_nonogram(grid, row_clues, col_clues, row_sequences, col_sequences, next_row, next_col)
            if result is not None:
                return result

        grid[row][col] = -1

    return None


def nonogram_solver(row_clues, col_clues):
    """
    Inicjuje i rozwiązuje nonogram dla zadanych wskazówek rzędów i kolumn.
    """
    height, width = len(row_clues), len(col_clues)
    grid = [[-1] * width for _ in range(height)]

    row_sequences = [generate_possible_sequences(clue, width) for clue in row_clues]
    col_sequences = [generate_possible_sequences(clue, height) for clue in col_clues]

    print("Wygenerowane sekwencje rzędów:")
    for i, seq in enumerate(row_sequences):
        print(f"Rząd {i + 1}: {seq}")

    print("Wygenerowane sekwencje kolumn:")
    for i, seq in enumerate(col_sequences):
        print(f"Kolumna {i + 1}: {seq}")

    solution = solve_nonogram(grid, row_clues, col_clues, row_sequences, col_sequences)

    if solution:
        return [" ".join(['#' if cell == 1 else '.' for cell in row]) for row in solution]
    else:
        return None
