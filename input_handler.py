def input_nonogram():
    """
    Pobiera dane od użytkownika dotyczące rozmiaru nonogramu oraz wskazówek.
    """
    sizes = [5, 10, 15, 20, 25, 30]
    print("Wybierz rozmiar nonogramu:")
    for i, size in enumerate(sizes):
        print(f"{i + 1}. {size}x{size}")

    choice = int(input("Podaj numer wyboru: ")) - 1
    size = sizes[choice]

    print(f"Wybrano nonogram {size}x{size}")

    row_clues = []
    col_clues = []

    print("Podaj wskazówki dla rzędów (oddzielone spacją, np. '3 1' dla 3, 1):")
    for i in range(size):
        while True:
            try:
                # Usuwamy nieoczekiwane znaki i wprowadzamy tylko liczby
                input_str = input(f"Rząd {i + 1}: ").replace(',', ' ').replace("'", '').strip()
                clues = list(map(int, input_str.split()))
                if not clues:
                    raise ValueError("Wartość nie może być pusta.")
                row_clues.append(clues)
                break
            except ValueError as e:
                print(f"Błąd: {e}. Spróbuj ponownie.")

    print("Podaj wskazówki dla kolumn (oddzielone spacją, np. '2 2' dla 2, 2):")
    for i in range(size):
        while True:
            try:
                # Usuwamy nieoczekiwane znaki i wprowadzamy tylko liczby
                input_str = input(f"Kolumna {i + 1}: ").replace(',', ' ').replace("'", '').strip()
                clues = list(map(int, input_str.split()))
                if not clues:
                    raise ValueError("Wartość nie może być pusta.")
                col_clues.append(clues)
                break
            except ValueError as e:
                print(f"Błąd: {e}. Spróbuj ponownie.")

    return row_clues, col_clues
