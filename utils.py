# utils.py

def generate_possible_sequences(clues, length):
    """
    Generuje wszystkie możliwe sekwencje dla podanych wskazówek i długości wiersza/kolumny.
    """
    if not clues:
        return [[0] * length]

    sequences = []
    total_blocks = sum(clues)
    total_gaps = len(clues) - 1
    total_empty = length - total_blocks - total_gaps

    def generate_sequences(current_sequence, remaining_clues, total_empty):
        if not remaining_clues:
            sequences.append(current_sequence + [0] * (length - len(current_sequence)))
            return
        current_clue = remaining_clues[0]
        remaining_clues = remaining_clues[1:]
        for i in range(total_empty + 1):
            new_sequence = current_sequence + [0] * i + [1] * current_clue
            generate_sequences(new_sequence + [0], remaining_clues, total_empty - i)

    generate_sequences([], clues, total_empty)
    return sequences


def is_valid_sequence(grid, sequence, index, is_row):
    """
    Sprawdza, czy dana sekwencja jest zgodna z aktualnym stanem rzędu lub kolumny.
    """
    line = grid[index] if is_row else [grid[i][index] for i in range(len(grid))]
    for i in range(len(line)):
        if line[i] != -1 and line[i] != int(sequence[i]):
            print(f"Niepoprawna sekwencja: {sequence} dla {'rzędu' if is_row else 'kolumny'} {index + 1}")
            print(f"Stan siatki: {line}")
            return False
    return True
