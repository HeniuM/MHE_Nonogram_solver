# MHE_Nonogram_solver

## Problem Nonogramu
### Opis problemu:
Nonogramy to logiczne łamigłówki, w których gracz musi wypełnić siatkę na podstawie wskazówek dotyczących liczby czarnych komórek w każdym rzędzie i kolumnie. Celem jest odkrycie ukrytego obrazu. <br>
<a href="https://en.wikipedia.org/wiki/Nonogram">Wikipedia</a>

### Implementacja problemu
1. Implementacja problemu znajduje się w przestrzeni nazw `MHE_Nonogram_solver`.
2. Do przechowywania wskazówek dotyczących rzędów i kolumn wykorzystano listy list, gdzie każda lista zawiera liczby oznaczające długości bloków czarnych komórek.
3. Główna funkcja rozwiązująca `nonogram_solver` znajduje się w pliku `solver.py` i wykorzystuje algorytm przeszukiwania z powrotem (backtracking).

### Algorytmy optymalizacyjne:
1. **Algorytm wspinaczkowy**:
   * Plik: `optimization_algorithms.py`
   * Funkcja: `hill_climbing(row_clues, col_clues)`
   * Wybiera najlepsze sąsiedztwo aktualnego rozwiązania i kontynuuje, aż nie znajdzie lepszego rozwiązania.

2. **Algorytm tabu**:
   * Plik: `optimization_algorithms.py`
   * Funkcja: `tabu_search(row_clues, col_clues, max_iter=1000, tabu_tenure=50)`
   * Wykorzystuje listę tabu do zapobiegania cyklom i lokalnym minimom.

3. **Algorytm symulowanego wyżarzania**:
   * Plik: `optimization_algorithms.py`
   * Funkcja: `simulated_annealing(row_clues, col_clues, initial_temp=100, cooling_rate=0.95, max_iter=1000)`
   * Wykorzystuje technikę chłodzenia do eksploracji przestrzeni rozwiązań.

### Algorytmy genetyczne:
1. **Algorytm genetyczny**:
   * Plik: `genetic_algorithms.py`
   * Funkcja: `genetic_algorithm(row_clues, col_clues, population_size=50, generations=100, mutation_rate=0.01)`
   * Wykorzystuje selekcję, krzyżowanie i mutację do generowania nowych populacji rozwiązań.

### Uruchamianie z linii komend:
1. Pobierz plik z wskazówkami dla rzędów oraz kolumn.
2. Uruchom wybrany algorytm, podając ścieżki do plików z wskazówkami jako argumenty.

### Przykład uruchomienia:
```bash
python optimization_algorithms.py hill_climbing row_clues.txt col_clues.txt
python optimization_algorithms.py tabu_search row_clues.txt col_clues.txt
python optimization_algorithms.py simulated_annealing row_clues.txt col_clues.txt
python genetic_algorithms.py row_clues.txt col_clues.txt
