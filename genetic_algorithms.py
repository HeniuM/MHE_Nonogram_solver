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
        sequences = generate_possible_sequences(clues, len(row))
        if row not in sequences:
            mismatches += 1

    for col_idx, clues in enumerate(col_clues):
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


def crossover_two_points(parent1, parent2):
    crossover_point1 = random.randint(1, len(parent1) - 2)
    crossover_point2 = random.randint(crossover_point1 + 1, len(parent1) - 1)
    child1 = parent1[:crossover_point1] + parent2[crossover_point1:crossover_point2] + parent1[crossover_point2:]
    child2 = parent2[:crossover_point1] + parent1[crossover_point1:crossover_point2] + parent2[crossover_point2:]
    return child1, child2


def mutate(solution, mutation_rate=0.05):
    mutated_solution = copy.deepcopy(solution)
    for i in range(len(mutated_solution)):
        for j in range(len(mutated_solution[0])):
            if random.random() < mutation_rate:
                mutated_solution[i][j] = 1 if mutated_solution[i][j] == 0 else 0
    return mutated_solution


def mutate_flip_bit(solution, mutation_rate=0.05):
    mutated_solution = copy.deepcopy(solution)
    num_mutations = int(mutation_rate * len(mutated_solution) * len(mutated_solution[0]))
    for _ in range(num_mutations):
        i = random.randint(0, len(mutated_solution) - 1)
        j = random.randint(0, len(mutated_solution[0]) - 1)
        mutated_solution[i][j] = 1 if mutated_solution[i][j] == 0 else 0
    return mutated_solution


def validate_solution(solution, row_clues, col_clues):
    for row, clues in zip(solution, row_clues):
        sequences = generate_possible_sequences(clues, len(row))
        if row not in sequences:
            return False

    for col_idx, clues in enumerate(col_clues):
        column = [solution[row_idx][col_idx] for row_idx in range(len(solution))]
        sequences = generate_possible_sequences(clues, len(column))
        if column not in sequences:
            return False

    return True


def get_best_solution(population, row_clues, col_clues):
    best_solution = population[0]
    best_score = evaluate_solution(best_solution, row_clues, col_clues)

    for individual in population:
        score = evaluate_solution(individual, row_clues, col_clues)
        if score < best_score:
            best_solution = individual
            best_score = score

    return best_solution, best_score


def sort_population(population, row_clues, col_clues):
    def sort_key(solution):
        return evaluate_solution(solution, row_clues, col_clues)

    return sorted(population, key=sort_key)


def genetic_algorithm(row_clues, col_clues, population_size=100, generations=200, mutation_rate=0.05, elitism_count=5):
    population = [generate_random_solution(row_clues, col_clues) for _ in range(population_size)]
    best_solution, best_score = get_best_solution(population, row_clues, col_clues)

    for _ in range(generations):
        population = sort_population(population, row_clues, col_clues)
        new_population = population[:elitism_count]

        while len(new_population) < population_size:
            parents = random.sample(population, 2)
            if random.random() < 0.5:
                child1, child2 = crossover(parents[0], parents[1])
            else:
                child1, child2 = crossover_two_points(parents[0], parents[1])
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))

        population = new_population

        current_best_solution, current_best_score = get_best_solution(population, row_clues, col_clues)

        if current_best_score < best_score:
            best_solution = current_best_solution
            best_score = current_best_score

    if best_score == 0 and validate_solution(best_solution, row_clues, col_clues):
        return best_solution
    else:
        return None


def parallel_genetic_algorithm(row_clues, col_clues, population_size=100, generations=200, mutation_rate=0.05,
                               elitism_count=5, num_workers=4):
    sub_population_size = population_size // num_workers
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for _ in range(num_workers):
            future = executor.submit(genetic_algorithm, row_clues, col_clues, sub_population_size, generations,
                                     mutation_rate, elitism_count)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    valid_results = [res for res in results if res is not None]

    if not valid_results:
        return None

    best_solution, best_score = get_best_solution(valid_results, row_clues, col_clues)

    for solution in valid_results:
        score = evaluate_solution(solution, row_clues, col_clues)
        if score < best_score:
            best_solution = solution
            best_score = score

    return best_solution


def island_genetic_algorithm(row_clues, col_clues, population_size=100, generations=200, mutation_rate=0.05,
                             elitism_count=5, num_islands=4, migration_rate=0.1, migration_interval=10):
    islands = []
    for _ in range(num_islands):
        island_population = []
        for _ in range(population_size // num_islands):
            island_population.append(generate_random_solution(row_clues, col_clues))
        islands.append(island_population)

    best_solution = None
    best_score = float('inf')

    for generation in range(generations):
        if generation % migration_interval == 0:
            for i in range(num_islands):
                if islands[i]:
                    for _ in range(int(population_size * migration_rate)):
                        migrant = random.choice(islands[i])
                        recipient_island = (i + 1) % num_islands
                        islands[recipient_island].append(migrant)

        for island_idx in range(num_islands):
            islands[island_idx] = sort_population(islands[island_idx], row_clues, col_clues)
            new_population = islands[island_idx][:elitism_count]

            while len(new_population) < population_size // num_islands:
                parents = random.sample(islands[island_idx], 2)
                if random.random() < 0.5:
                    child1, child2 = crossover(parents[0], parents[1])
                else:
                    child1, child2 = crossover_two_points(parents[0], parents[1])
                new_population.append(mutate(child1, mutation_rate))
                new_population.append(mutate(child2, mutation_rate))

            islands[island_idx] = new_population
            current_best_solution, current_best_score = get_best_solution(islands[island_idx], row_clues, col_clues)

            if current_best_score < best_score:
                best_solution = current_best_solution
                best_score = current_best_score

    if best_score == 0 and validate_solution(best_solution, row_clues, col_clues):
        return best_solution
    else:
        return None


def evolutionary_strategy(row_clues, col_clues, population_size=100, generations=200, mutation_rate=0.05,
                          elitism_count=5):
    population = [generate_random_solution(row_clues, col_clues) for _ in range(population_size)]
    best_solution, best_score = get_best_solution(population, row_clues, col_clues)

    for _ in range(generations):
        new_population = []

        for individual in population:
            new_individual = mutate(individual, mutation_rate)
            new_population.append(new_individual)

        population = sort_population(new_population + population, row_clues, col_clues)[:population_size]

        current_best_solution, current_best_score = get_best_solution(population, row_clues, col_clues)

        if current_best_score < best_score:
            best_solution = current_best_solution
            best_score = current_best_score

    if best_score == 0 and validate_solution(best_solution, row_clues, col_clues):
        return best_solution
    else:
        return None


def run_genetic_algorithms(row_clues, col_clues):
    print("Running Genetic Algorithm 1...")
    ga_solution_1 = genetic_algorithm(row_clues, col_clues)

    print("Running Parallel Genetic Algorithm...")
    ga_solution_2 = parallel_genetic_algorithm(row_clues, col_clues)

    print("Running Island Genetic Algorithm...")
    ga_solution_3 = island_genetic_algorithm(row_clues, col_clues)

    print("Running Evolutionary Strategy Algorithm...")
    ga_solution_4 = evolutionary_strategy(row_clues, col_clues)

    return ga_solution_1, ga_solution_2, ga_solution_3, ga_solution_4
