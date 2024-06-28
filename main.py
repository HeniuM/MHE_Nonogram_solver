from input_handler import input_nonogram
from solver import nonogram_solver


def main():
    row_clues, col_clues = input_nonogram()
    solution = nonogram_solver(row_clues, col_clues)

    if solution:
        for row in solution:
            print(row)
    else:
        print("Nie znaleziono rozwiązania.")


if __name__ == "__main__":
    main()
