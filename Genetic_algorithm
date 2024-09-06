# Part 1


import random

# Initialize population with random chromosomes
def initialize_population(pop_size, N, T):
    population = []
    for _ in range(pop_size):
        chromosome = ['0'] * (N * T)
        for n in range(N):
            timeslot = random.randint(0, T - 1)
            chromosome[timeslot * N + n] = '1'
        population.append(''.join(chromosome))
    return population

# Calculate fitness by evaluating overlap and consistency penalties
def calculate_fitness(chromosome, N, T):
    overlap_penalty = 0
    consistency_penalty = 0
    
    # Calculate overlap penalty
    for t in range(T):
        timeslot = chromosome[t * N: (t + 1) * N]
        overlap_penalty += max(0, sum(map(int, timeslot)) - 1)

    # Calculate consistency penalty
    for n in range(N):
        course_schedule = chromosome[n::N]
        occurrences = course_schedule.count('1')
        consistency_penalty += abs(occurrences - 1)
    
    return -(overlap_penalty + consistency_penalty)

# Select two parents based on their fitness
def select_parents(population, fitnesses):
    total_fitness = sum(fitnesses)
    selection_probs = [fitness / total_fitness for fitness in fitnesses]
    parents = random.choices(population, weights=selection_probs, k=2)
    return parents

# Perform single-point crossover
def single_point_crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    offspring1 = parent1[:point] + parent2[point:]
    offspring2 = parent2[:point] + parent1[point:]
    return offspring1, offspring2

# Mutate a chromosome with a given mutation rate
def mutate(chromosome, mutation_rate):
    chromosome = list(chromosome)
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = '1' if chromosome[i] == '0' else '0'
    return ''.join(chromosome)

# Genetic algorithm to find the optimal course schedule
def genetic_algorithm(N, T, courses, pop_size=100, max_generations=1000, mutation_rate=0.01):
    population = initialize_population(pop_size, N, T)
    best_fitness = float('-inf')
    best_chromosome = None
    
    for generation in range(max_generations):
        fitnesses = [calculate_fitness(chrom, N, T) for chrom in population]
        
        for i in range(pop_size):
            if fitnesses[i] > best_fitness:
                best_fitness = fitnesses[i]
                best_chromosome = population[i]
        
        if best_fitness == 0:
            break
        
        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = select_parents(population, fitnesses)
            offspring1, offspring2 = single_point_crossover(parent1, parent2)
            offspring1 = mutate(offspring1, mutation_rate)
            offspring2 = mutate(offspring2, mutation_rate)
            new_population.extend([offspring1, offspring2])
        
        population = new_population[:pop_size]
    
    return best_chromosome, best_fitness

def read_course_data(filename):
    with open(filename, 'r') as file:
        N, T = map(int, file.readline().split())
        courses = [file.readline().strip() for _ in range(N)]
    return N, T, courses

# Example usage:
filename = '/content/input.txt'
N, T, courses = read_course_data(filename)

best_chromosome, best_fitness = genetic_algorithm(N, T, courses)
print("Best Chromosome:", best_chromosome)
print("Best Fitness:", best_fitness)



# Part 2

import random

# Initialize population with random chromosomes
def initialize_population(pop_size, N, T):
    population = []
    for _ in range(pop_size):
        chromosome = ['0'] * (N * T)
        for n in range(N):
            timeslot = random.randint(0, T - 1)
            chromosome[timeslot * N + n] = '1'
        population.append(''.join(chromosome))
    return population

# Two-point crossover function
def two_point_crossover(parent1, parent2):
    point1 = random.randint(1, len(parent1) - 2)
    point2 = random.randint(point1 + 1, len(parent1) - 1)
    
    offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    
    return offspring1, offspring2

# Example Usage
N = 3
T = 3
pop_size = 100

# Initialize population
population = initialize_population(pop_size, N, T)

# Randomly select two parents from the population
parent1, parent2 = random.sample(population, 2)

# Perform two-point crossover
offspring1, offspring2 = two_point_crossover(parent1, parent2)

# Print the parents and the offspring
print("Parent 1: ", parent1)
print("Parent 2: ", parent2)
print("Offspring 1: ", offspring1)
print("Offspring 2: ", offspring2)



# Part 3


import random

# Initialize population with random chromosomes
def initialize_population(pop_size, N, T):
    population = []
    for _ in range(pop_size):
        chromosome = ['0'] * (N * T)
        for n in range(N):
            timeslot = random.randint(0, T - 1)
            chromosome[timeslot * N + n] = '1'
        population.append(''.join(chromosome))
    return population

# Calculate fitness by evaluating overlap and consistency penalties
def calculate_fitness(chromosome, N, T):
    overlap_penalty = 0
    consistency_penalty = 0
    
    # Calculate overlap penalty
    for t in range(T):
        timeslot = chromosome[t * N: (t + 1) * N]
        overlap_penalty += max(0, sum(map(int, timeslot)) - 1)

    # Calculate consistency penalty
    for n in range(N):
        course_schedule = chromosome[n::N]
        occurrences = course_schedule.count('1')
        consistency_penalty += abs(occurrences - 1)
    
    return -(overlap_penalty + consistency_penalty)

# Tournament Selection
def tournament_selection(population, fitnesses, k=3):
    selected = random.sample(list(zip(population, fitnesses)), k)
    selected = sorted(selected, key=lambda x: x[1], reverse=True)
    return selected[0][0], selected[1][0]

# Perform single-point crossover
def single_point_crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    offspring1 = parent1[:point] + parent2[point:]
    offspring2 = parent2[:point] + parent1[point:]
    return offspring1, offspring2

# Mutate a chromosome with a given mutation rate
def mutate(chromosome, mutation_rate):
    chromosome = list(chromosome)
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = '1' if chromosome[i] == '0' else '0'
    return ''.join(chromosome)

# Genetic algorithm to find the optimal course schedule
def genetic_algorithm(N, T, courses, pop_size=100, max_generations=1000, mutation_rate=0.01):
    population = initialize_population(pop_size, N, T)
    best_fitness = float('-inf')
    best_chromosome = None
    
    for generation in range(max_generations):
        fitnesses = [calculate_fitness(chrom, N, T) for chrom in population]
        
        for i in range(pop_size):
            if fitnesses[i] > best_fitness:
                best_fitness = fitnesses[i]
                best_chromosome = population[i]
        
        if best_fitness == 0:
            break
        
        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = tournament_selection(population, fitnesses)
            offspring1, offspring2 = single_point_crossover(parent1, parent2)
            offspring1 = mutate(offspring1, mutation_rate)
            offspring2 = mutate(offspring2, mutation_rate)
            new_population.extend([offspring1, offspring2])
        
        population = new_population[:pop_size]
    
    return best_chromosome, best_fitness

def read_course_data(filename):
    with open(filename, 'r') as file:
        N, T = map(int, file.readline().split())
        courses = [file.readline().strip() for _ in range(N)]
    return N, T, courses

# Example usage:
filename = '/content/input.txt'
N, T, courses = read_course_data(filename)

best_chromosome, best_fitness = genetic_algorithm(N, T, courses)
print("Best Chromosome:", best_chromosome)
print("Best Fitness:", best_fitness)



