import random
from typing import List
from classes import Item, Individual, items, MAX_KNAPSACK_WEIGHT, MUTATION_RATE, CROSSOVER_RATE, REPRODUCTION_RATE


def generate_initial_population(count=6) -> List[Individual]:
    population = set()

    # generate initial population having `count` individuals
    while len(population) != count:
        # pick random bits one for each item and 
        # create an individual 
        bits = [
            random.choice([0, 1])
            for _ in items
        ]
        population.add(Individual(bits))

    return list(population)


def selection(population: List[Individual]) -> List[Individual]:
    parents = []
    
    # randomly shuffle the population
    random.shuffle(population)

    # we use the first 4 individuals
    # run a tournament between them and
    # get two fit parents for the next steps of evolution

    # tournament between first and second
    if population[0].fitness() > population[1].fitness():
        parents.append(population[0])
    else:
        parents.append(population[1])
    
    # tournament between third and fourth
    if population[2].fitness() > population[3].fitness():
        parents.append(population[2])
    else:
        parents.append(population[3])

    return parents


def crossover(parents: List[Individual]) -> List[Individual]:
    N = len(items)

    child1 = parents[0].bits[:N//2] + parents[1].bits[N//2:]
    child2 = parents[0].bits[N//2:] + parents[1].bits[:N//2]

    return [Individual(child1), Individual(child2)]


def mutate(individuals: List[Individual]) -> List[Individual]:
    for individual in individuals:
        for i in range(len(individual.bits)):
            if random.random() < MUTATION_RATE:
                # Flip the bit
                individual.bits[i] = ~individual.bits[i]


def next_generation(population: List[Individual]) -> List[Individual]:
    next_gen = []
    while len(next_gen) < len(population):
        children = []

        # we run selection and get parents
        parents = selection(population)

        # reproduction
        if random.random() < REPRODUCTION_RATE:
            children = parents
        else:
            # crossover
            if random.random() < CROSSOVER_RATE:
                children = crossover(parents)
            
            # mutation
            if random.random() < MUTATION_RATE:
                mutate(children)

        next_gen.extend(children)

    return next_gen[:len(population)]


def print_generation(population: List[Individual], sorted: bool = False, detailed: bool = False):
    if sorted:
        population.sort(key=lambda x: x.fitness())
    if not detailed:
        for individual in population:
            print(individual.bits, individual.fitness())
        print()
        print("Average fitness", sum([x.fitness() for x in population])/len(population))
        print("-" * 32)
    else:
        accepted = list(filter(lambda x: x.fitness() != 0,population))
        rejected = list(filter(lambda x: x.fitness() == 0,population))
        print("=" * 5 + " Accepted solutions " + "="*5)
        for element in accepted:
            print(element.bits, element.fitness())
        print("=" * 5 + " Rejected solutions " + "="*5)
        for element in rejected:
            print(element.bits, element.fitness())
        print()
        print("Average fitness", sum([x.fitness() for x in accepted])/len(accepted))
        print("-" * 32)       

def average_fitness(population: List[Individual]) -> float:
    return sum([i.fitness() for i in population]) / len(population)


def solve_knapsack() -> Individual:
    population = generate_initial_population()

    avg_fitnesses = []

    for _ in range(500):
        avg_fitnesses.append(average_fitness(population))
        population = next_generation(population)

    population = sorted(population, key=lambda i: i.fitness(), reverse=True)
    return population[0]

