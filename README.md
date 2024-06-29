# MHE_Nonogram_solver

## Problem Nonogramu
### Opis problemu:
Nonogramy to logiczne łamigłówki, w których gracz musi wypełnić siatkę na podstawie wskazówek dotyczących liczby czarnych komórek w każdym rzędzie i kolumnie. Celem jest odkrycie ukrytego obrazu.
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
   * Funkcja: `tabu_search(row_clues, col_clues, max_iterations=100, tabu_tenure=5)`
   * Wykorzystuje listę tabu do zapobiegania cyklom i lokalnym minimom.

3. **Algorytm symulowanego wyżarzania**:
   * Plik: `optimization_algorithms.py`
   * Funkcja: `simulated_annealing(row_clues, col_clues, initial_temp=100, cooling_rate=0.95, max_iterations=1000)`
   * Wykorzystuje technikę chłodzenia do eksploracji przestrzeni rozwiązań.

### Algorytmy genetyczne:
1. **Algorytm genetyczny**:
   * Plik: `genetic_algorithms.py`
   * Funkcja: `genetic_algorithm(row_clues, col_clues, population_size=100, generations=200, mutation_rate=0.05, elitism_count=5)`
   * Wykorzystuje selekcję, krzyżowanie i mutację do generowania nowych populacji rozwiązań.

2. **Równoległy algorytm genetyczny**:
   * Plik: `genetic_algorithms.py`
   * Funkcja: `parallel_genetic_algorithm(row_clues, col_clues, population_size=100, generations=200, mutation_rate=0.05, elitism_count=5, num_workers=4)`
   * Wykorzystuje równoległe przetwarzanie do przyspieszenia procesu ewolucji.

3. **Algorytm genetyczny z wyspami**:
   * Plik: `genetic_algorithms.py`
   * Funkcja: `island_genetic_algorithm(row_clues, col_clues, population_size=100, generations=200, mutation_rate=0.05, elitism_count=5, num_islands=4, migration_rate=0.1, migration_interval=10)`
   * Wykorzystuje koncepcję wysp i migracji do zwiększenia różnorodności genetycznej.

4. **Strategia ewolucyjna**:
   * Plik: `genetic_algorithms.py`
   * Funkcja: `evolutionary_strategy(row_clues, col_clues, population_size=100, generations=200, mutation_rate=0.05, elitism_count=5)`
   * Wykorzystuje mutacje do generowania nowych populacji rozwiązań.

### Uruchamianie z linii komend:
1. Pobierz plik z wskazówkami dla rzędów oraz kolumn.
2. Uruchom wybrany algorytm, podając ścieżki do plików z wskazówkami jako argumenty.

### Przykład uruchomienia:
```bash
# Manualne wprowadzanie danych
python main.py --input-method manual

# Wprowadzanie danych z plików
python main.py --input-method file --row-clues-file row_clues.txt --col-clues-file col_clues.txt

# Algorytmy optymalizacyjne
python main.py --input-method file --row-clues-file row_clues.txt --col-clues-file col_clues.txt --algorithm hill_climbing
python main.py --input-method file --row-clues-file row_clues.txt --col-clues-file col_clues.txt --algorithm tabu_search
python main.py --input-method file --row-clues-file row_clues.txt --col-clues-file col_clues.txt --algorithm simulated_annealing

# Algorytmy genetyczne
python main.py --input-method file --row-clues-file row_clues.txt --col-clues-file col_clues.txt --algorithm genetic_algorithm
python main.py --input-method file --row-clues-file row_clues.txt --col-clues-file col_clues.txt --algorithm parallel_genetic_algorithm
python main.py --input-method file --row-clues-file row_clues.txt --col-clues-file col_clues.txt --algorithm island_genetic_algorithm
python main.py --input-method file --row-clues-file row_clues.txt --col-clues-file col_clues.txt --algorithm evolutionary_strategy


