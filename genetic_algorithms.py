import random
import copy
import concurrent.futures


def generate_random_solution(row_clues, col_clues):
    height, width = len(row_clues), len(col_clues)
    solution = [[random.choice([0, 1]) for _ in range(width)] for _ in range(height)]
    return solution


def generate_possible_sequences(clues, length):
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
            if remaining_clues:
                new_sequence += [0]
            generate_sequences(new_sequence, remaining_clues, total_empty - i)

    generate_sequences([], clues, total_empty)
    return sequences


def evaluate_solution(grid, row_clues, col_clues):
    mismatches = 0

    for row, clues in zip(grid, row_clues):
        if isinstance(clues, int):  # Dodanie sprawdzania typu
            clues = [clues]
        sequences = generate_possible_sequences(clues, len(row))
        if row not in sequences:
            mismatches += 1

    for col_idx, clues in enumerate(col_clues):
        if isinstance(clues, int):  # Dodanie sprawdzania typu
            clues = [clues]
        column = [grid[row_idx][col_idx] for row_idx in range(len(grid))]
        sequences = generate_possible_sequences(clues, len(column))
        if column not in sequences:
            mismatches += 1

    return mismatches


def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(solution, mutation_rate=0.05):
    mutated_solution = copy.deepcopy(solution)
    for i in range(len(mutated_solution)):
        for j in range(len(mutated_solution[0])):
            if random.random() < mutation_rate:
                mutated_solution[i][j] = 1 if mutated_solution[i][j] == 0 else 0
    return mutated_solution


def validate_solution(solution, row_clues, col_clues):
    for row, clues in zip(solution, row_clues):
        if isinstance(clues, int):  # Dodanie sprawdzania typu
            clues = [clues]
        sequences = generate_possible_sequences(clues, len(row))
        if row not in sequences:
            return False

    for col_idx, clues in enumerate(col_clues):
        if isinstance(clues, int):  # Dodanie sprawdzania typu
            clues = [clues]
        column = [solution[row_idx][col_idx] for row_idx in range(len(solution))]
        sequences = generate_possible_sequences(clues, len(column))
        if column not in sequences:
            return False

    return True


def genetic_algorithm(row_clues, col_clues, population_size=100, generations=200, mutation_rate=0.05, elitism_count=5):
    population = [generate_random_solution(row_clues, col_clues) for _ in range(population_size)]
    best_solution = min(population, key=lambda sol: evaluate_solution(sol, row_clues, col_clues))
    best_score = evaluate_solution(best_solution, row_clues, col_clues)

    for generation in range(generations):
        print(f"Generation {generation}: Best Score = {best_score}")

        population = sorted(population, key=lambda sol: evaluate_solution(sol, row_clues, col_clues))
        new_population = population[:elitism_count]

        while len(new_population) < population_size:
            parents = random.sample(population, 2)
            child1, child2 = crossover(parents[0], parents[1])
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))

        population = sorted(new_population, key=lambda sol: evaluate_solution(sol, row_clues, col_clues))[
                     :population_size]

        current_best_solution = population[0]
        current_best_score = evaluate_solution(current_best_solution, row_clues, col_clues)

        if current_best_score < best_score:
            best_solution, best_score = current_best_solution, current_best_score

        # Logging current population and best solution for debugging
        print(f"Best solution in generation {generation}: {current_best_solution} with score {current_best_score}")

    return best_solution if best_score == 0 and validate_solution(best_solution, row_clues, col_clues) else None


def worker_func(row_clues, col_clues, sub_population_size, generations, mutation_rate, elitism_count):
    return genetic_algorithm(row_clues, col_clues, sub_population_size, generations, mutation_rate, elitism_count)


def parallel_genetic_algorithm(row_clues, col_clues, population_size=100, generations=200, mutation_rate=0.05,
                               elitism_count=5, num_workers=4):
    sub_population_size = population_size // num_workers

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_func, row_clues, col_clues, sub_population_size, generations, mutation_rate,
                                   elitism_count)
                   for _ in range(num_workers)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    valid_results = [res for res in results if res is not None]

    if not valid_results:
        return None

    best_solution = min(valid_results,
                        key=lambda sol: evaluate_solution(sol, row_clues, col_clues) if sol else float('inf'))
    return best_solution


def island_genetic_algorithm(row_clues, col_clues, population_size=100, generations=200, mutation_rate=0.05,
                             elitism_count=5, num_islands=4, migration_rate=0.1, migration_interval=10):
    def genetic_algorithm_with_args():
        return genetic_algorithm(row_clues, col_clues, population_size // num_islands, generations // num_islands,
                                 mutation_rate, elitism_count)

    islands = [[generate_random_solution(row_clues, col_clues) for _ in range(population_size // num_islands)] for _ in
               range(num_islands)]
    best_solution = None
    best_score = float('inf')

    for generation in range(generations):
        print(f"Island Genetic Algorithm - Generation {generation}")

        # Perform migration every `migration_interval` generations
        if generation % migration_interval == 0:
            print(f"Migration at generation {generation}")
            for i in range(num_islands):
                if islands[i]:
                    for _ in range(int(population_size * migration_rate)):
                        migrant = random.choice(islands[i])
                        recipient_island = (i + 1) % num_islands
                        islands[recipient_island].append(migrant)

        for island_idx in range(num_islands):
            islands[island_idx] = sorted(islands[island_idx],
                                         key=lambda sol: evaluate_solution(sol, row_clues, col_clues))
            new_population = islands[island_idx][:elitism_count]

            while len(new_population) < population_size // num_islands:
                parents = random.sample(islands[island_idx], 2)
                child1, child2 = crossover(parents[0], parents[1])
                new_population.append(mutate(child1, mutation_rate))
                new_population.append(mutate(child2, mutation_rate))

            islands[island_idx] = new_population
            current_best_solution = islands[island_idx][0]
            current_best_score = evaluate_solution(current_best_solution, row_clues, col_clues)

            if current_best_score < best_score:
                best_solution, best_score = current_best_solution, current_best_score

        print(f"Best solution on all islands in generation {generation}: {best_solution} with score {best_score}")

    return best_solution if best_score == 0 and validate_solution(best_solution, row_clues, col_clues) else None


def evolutionary_strategy(row_clues, col_clues, population_size=100, generations=200, mutation_rate=0.05,
                          elitism_count=5):
    population = [generate_random_solution(row_clues, col_clues) for _ in range(population_size)]
    best_solution = min(population, key=lambda sol: evaluate_solution(sol, row_clues, col_clues))
    best_score = evaluate_solution(best_solution, row_clues, col_clues)

    for generation in range(generations):
        population = sorted(population, key=lambda sol: evaluate_solution(sol, row_clues, col_clues))
        new_population = population[:elitism_count]

        while len(new_population) < population_size:
            parents = random.sample(population, 2)
            child1, child2 = crossover(parents[0], parents[1])
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))

        population = sorted(new_population, key=lambda sol: evaluate_solution(sol, row_clues, col_clues))[
                     :population_size]

        current_best_solution = population[0]
        current_best_score = evaluate_solution(current_best_solution, row_clues, col_clues)

        if current_best_score < best_score:
            best_solution, best_score = current_best_solution, current_best_score

        # Logging current population and best solution for debugging
        print(f"Best solution in generation {generation}: {current_best_solution} with score {current_best_score}")

    return best_solution if best_score == 0 and validate_solution(best_solution, row_clues, col_clues) else None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Nonogram Solver using Genetic Algorithm")
    parser.add_argument("row_clues_file", help="File with row clues")
    parser.add_argument("col_clues_file", help="File with column clues")
    parser.add_argument("algorithm",
                        help="Algorithm to use: genetic, parallel_genetic, island_genetic, evolutionary_strategy")

    args = parser.parse_args()

    with open(args.row_clues_file, 'r') as file:
        row_clues = [list(map(int, line.strip().split())) for line in file]

    with open(args.col_clues_file, 'r') as file:
        col_clues = [list(map(int, line.strip().split())) for line in file]

    if args.algorithm == "genetic":
        solution = genetic_algorithm(row_clues, col_clues)
    elif args.algorithm == "parallel_genetic":
        solution = parallel_genetic_algorithm(row_clues, col_clues)
    elif args.algorithm == "island_genetic":
        solution = island_genetic_algorithm(row_clues, col_clues)
    elif args.algorithm == "evolutionary_strategy":
        solution = evolutionary_strategy(row_clues, col_clues)
    else:
        print(f"Unknown algorithm: {args.algorithm}")
        solution = None

    if solution:
        for row in solution:
            print(" ".join(['#' if cell == 1 else '.' for cell in row]))
    else:
        print("Nie znaleziono rozwiązania.")
