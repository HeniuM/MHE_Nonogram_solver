import random
import numpy as np


def generate_sequences(clues, length):
    if not clues:
        return [[0] * length]

    sequences = []
    first_clue = clues[0]
    for start in range(length - first_clue + 1):
        prefix = [0] * start + [1] * first_clue
        remaining_length = length - len(prefix)
        if len(clues) == 1:
            suffixes = [[0] * remaining_length]
        else:
            suffixes = generate_sequences(clues[1:], remaining_length - 1)
            suffixes = [[0] + suffix for suffix in suffixes]
        for suffix in suffixes:
            sequences.append(prefix + suffix)
    return sequences


def generate_initial_solution(row_clues, col_clues):
    solution = []
    for clues in row_clues:
        sequences = generate_sequences(clues, len(col_clues))
        solution.append(random.choice(sequences))
    return solution


def fitness(solution, row_clues, col_clues):
    fitness_score = 0
    for row, clues in zip(solution, row_clues):
        sequences = generate_sequences(clues, len(row))
        if row in sequences:
            fitness_score += 1
    for col_idx, clues in enumerate(col_clues):
        column = [solution[row_idx][col_idx] for row_idx in range(len(solution))]
        sequences = generate_sequences(clues, len(column))
        if column in sequences:
            fitness_score += 1
    return fitness_score


def hill_climbing(row_clues, col_clues):
    solution = generate_initial_solution(row_clues, col_clues)
    current_fitness = fitness(solution, row_clues, col_clues)
    while True:
        neighbors = []
        for row_idx, row in enumerate(solution):
            for sequence in generate_sequences(row_clues[row_idx], len(row)):
                if sequence != row:
                    neighbor = solution[:]
                    neighbor[row_idx] = sequence
                    neighbors.append(neighbor)
        best_neighbor = neighbors[0]
        best_fitness = fitness(neighbor, row_clues, col_clues)
        for neighbor in neighbors:
            if fitness(neighbor, row_clues, col_clues) > best_fitness:
                best_neighbor = neighbor
                best_fitness = fitness(neighbor, row_clues, col_clues)
        if best_fitness <= current_fitness:
            break
        solution, current_fitness = best_neighbor, best_fitness
    return solution


def tabu_search(row_clues, col_clues, tabu_tenure=5, max_iterations=100):
    solution = generate_initial_solution(row_clues, col_clues)
    current_fitness = fitness(solution, row_clues, col_clues)
    tabu_list = []
    best_solution = solution[:]
    best_fitness = current_fitness

    for _ in range(max_iterations):
        neighbors = []
        for row_idx, row in enumerate(solution):
            for sequence in generate_sequences(row_clues[row_idx], len(row)):
                if sequence != row:
                    neighbor = solution[:]
                    neighbor[row_idx] = sequence
                    if neighbor not in tabu_list:
                        neighbors.append(neighbor)

        best_neighbor = neighbors[0]
        best_neighbor_fitness = fitness(best_neighbor, row_clues, col_clues)
        for neighbor in neighbors:
            if fitness(neighbor, row_clues, col_clues) > best_neighbor_fitness:
                best_neighbor = neighbor
                best_neighbor_fitness = fitness(neighbor, row_clues, col_clues)

        if best_neighbor_fitness > best_fitness:
            best_solution = best_neighbor[:]
            best_fitness = best_neighbor_fitness

        solution = best_neighbor[:]
        current_fitness = best_neighbor_fitness
        tabu_list.append(solution)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

    return best_solution


def simulated_annealing(row_clues, col_clues, initial_temp=100, cooling_rate=0.95, max_iterations=1000):
    solution = generate_initial_solution(row_clues, col_clues)
    current_fitness = fitness(solution, row_clues, col_clues)
    best_solution = solution[:]
    best_fitness = current_fitness
    temperature = initial_temp

    for _ in range(max_iterations):
        neighbor = solution[:]
        row_idx = random.randint(0, len(solution) - 1)
        sequence = random.choice(generate_sequences(row_clues[row_idx], len(solution[0])))
        neighbor[row_idx] = sequence
        neighbor_fitness = fitness(neighbor, row_clues, col_clues)

        if neighbor_fitness > current_fitness or random.random() < np.exp(
                (neighbor_fitness - current_fitness) / temperature):
            solution = neighbor[:]
            current_fitness = neighbor_fitness

        if current_fitness > best_fitness:
            best_solution = solution[:]
            best_fitness = current_fitness

        temperature *= cooling_rate

    return best_solution


def run_optimization_algorithms(row_clues, col_clues):
    print("Running Hill Climbing Algorithm...")
    hc_solution = hill_climbing(row_clues, col_clues)

    print("Running Tabu Search Algorithm...")
    ts_solution = tabu_search(row_clues, col_clues)

    print("Running Simulated Annealing Algorithm...")
    sa_solution = simulated_annealing(row_clues, col_clues)

    return hc_solution, ts_solution, sa_solution
