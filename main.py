import argparse
import time
import psutil
from input_handler import input_nonogram
from solver import nonogram_solver
from genetic_algorithms import genetic_algorithm, parallel_genetic_algorithm, island_genetic_algorithm, \
    evolutionary_strategy
from optimization_algorithms import hill_climbing, tabu_search, simulated_annealing


def display_solution(name, solution, exec_time, memory):
    if solution:
        print(f"{name} Solution:")
        for row in solution:
            print("".join(['#' if cell == 1 else '.' for cell in row]))
        print(f"{name} Execution Time: {exec_time:.2f} seconds")
        print(f"{name} Memory Usage: {memory / (1024 * 1024):.2f} MB\n")
    else:
        print(f"{name} did not find a solution.\n")


def measure_performance(func, *args):
    process = psutil.Process()
    start_time = time.time()
    start_memory = process.memory_info().rss
    result = func(*args)
    end_time = time.time()
    end_memory = process.memory_info().rss
    execution_time = end_time - start_time
    memory_usage = end_memory - start_memory
    return result, execution_time, memory_usage


def get_best_time(results):
    best_time = results[0]
    for result in results:
        if result[1] < best_time[1]:
            best_time = result
    return best_time


def get_best_memory(results):
    best_memory = results[0]
    for result in results:
        if result[2] < best_memory[2]:
            best_memory = result
    return best_memory


def run_multiple_times(func, row_clues, col_clues, runs=3):
    best_result = None
    best_time = float('inf')
    best_memory = float('inf')
    for _ in range(runs):
        result, exec_time, memory = measure_performance(func, row_clues, col_clues)
        if result and exec_time < best_time:
            best_result = result
            best_time = exec_time
            best_memory = memory
    return best_result, best_time, best_memory


def main():
    parser = argparse.ArgumentParser(description="Nonogram Solver")
    parser.add_argument('--input-method', choices=['manual', 'file'], required=True,
                        help='Input method for nonogram clues')
    parser.add_argument('--row-clues-file', type=str, help='File with row clues')
    parser.add_argument('--col-clues-file', type=str, help='File with column clues')
    parser.add_argument('--algorithm',
                        choices=['all', 'brute_force', 'genetic', 'hill_climbing', 'tabu_search', 'simulated_annealing',
                                 'genetic_algorithm', 'parallel_genetic_algorithm', 'island_genetic_algorithm',
                                 'evolutionary_strategy'],
                        default='all', help='Algorithm to run')
    args = parser.parse_args()

    if args.input_method == 'manual':
        row_clues, col_clues = input_nonogram()
    elif args.input_method == 'file':
        if not args.row_clues_file or not args.col_clues_file:
            parser.error("--row-clues-file and --col-clues-file are required when using file input method")
        with open(args.row_clues_file, 'r') as file:
            row_clues = [list(map(int, line.strip().split())) for line in file]
        with open(args.col_clues_file, 'r') as file:
            col_clues = [list(map(int, line.strip().split())) for line in file]

    results = []

    if args.algorithm in ['all', 'brute_force']:
        print("Running Brute Force Algorithm...")
        bf_solution, bf_time, bf_memory = measure_performance(nonogram_solver, row_clues, col_clues)
        display_solution("Brute Force", bf_solution, bf_time, bf_memory)
        results.append(("Brute Force", bf_time, bf_memory))

    if args.algorithm in ['all', 'genetic', 'genetic_algorithm']:
        print("Running Genetic Algorithm...")
        ga_solution_1, ga_time_1, ga_memory_1 = run_multiple_times(genetic_algorithm, row_clues, col_clues)
        display_solution("Genetic Algorithm", ga_solution_1, ga_time_1, ga_memory_1)
        results.append(("Genetic Algorithm", ga_time_1, ga_memory_1))

    if args.algorithm in ['all', 'genetic', 'parallel_genetic_algorithm']:
        print("Running Parallel Genetic Algorithm...")
        ga_solution_2, ga_time_2, ga_memory_2 = run_multiple_times(parallel_genetic_algorithm, row_clues, col_clues)
        display_solution("Parallel Genetic Algorithm", ga_solution_2, ga_time_2, ga_memory_2)
        results.append(("Parallel Genetic Algorithm", ga_time_2, ga_memory_2))

    if args.algorithm in ['all', 'genetic', 'island_genetic_algorithm']:
        print("Running Island Genetic Algorithm...")
        ga_solution_3, ga_time_3, ga_memory_3 = run_multiple_times(island_genetic_algorithm, row_clues, col_clues)
        display_solution("Island Genetic Algorithm", ga_solution_3, ga_time_3, ga_memory_3)
        results.append(("Island Genetic Algorithm", ga_time_3, ga_memory_3))

    if args.algorithm in ['all', 'genetic', 'evolutionary_strategy']:
        print("Running Evolutionary Strategy Algorithm...")
        ga_solution_4, ga_time_4, ga_memory_4 = run_multiple_times(evolutionary_strategy, row_clues, col_clues)
        display_solution("Evolutionary Strategy", ga_solution_4, ga_time_4, ga_memory_4)
        results.append(("Evolutionary Strategy", ga_time_4, ga_memory_4))

    if args.algorithm in ['all', 'hill_climbing']:
        print("Running Hill Climbing Algorithm...")
        hc_solution, hc_time, hc_memory = measure_performance(hill_climbing, row_clues, col_clues)
        display_solution("Hill Climbing", hc_solution, hc_time, hc_memory)
        results.append(("Hill Climbing", hc_time, hc_memory))

    if args.algorithm in ['all', 'tabu_search']:
        print("Running Tabu Search Algorithm...")
        ts_solution, ts_time, ts_memory = measure_performance(tabu_search, row_clues, col_clues)
        display_solution("Tabu Search", ts_solution, ts_time, ts_memory)
        results.append(("Tabu Search", ts_time, ts_memory))

    if args.algorithm in ['all', 'simulated_annealing']:
        print("Running Simulated Annealing Algorithm...")
        sa_solution, sa_time, sa_memory = measure_performance(simulated_annealing, row_clues, col_clues)
        display_solution("Simulated Annealing", sa_solution, sa_time, sa_memory)
        results.append(("Simulated Annealing", sa_time, sa_memory))

    if args.algorithm == 'all':
        best_time = get_best_time(results)
        best_memory = get_best_memory(results)

        print("\nSummary:\n")
        print(f"Fastest Algorithm: {best_time[0]} with Execution Time: {best_time[1]:.2f} seconds")
        print(
            f"Most Memory Efficient Algorithm: {best_memory[0]} with Memory Usage: {best_memory[2] / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
