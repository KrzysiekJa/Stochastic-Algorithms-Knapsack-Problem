import random
from typing import List
import numpy as np
import matplotlib.pyplot as plt


class Item:
    def __init__(self, name, weight, value):
        self.name = name
        self.weight = weight
        self.value = value


class Individual:
    def __init__(self, bits: List[int]):
        self.bits = bits
    
    def __str__(self):
        return repr(self.bits)

    def __hash__(self):
        return hash(str(self.bits))
    
    def fitness(self) -> float:
        """
        total_value = sum([
            bit * item.value
            for item, bit in zip(items, self.bits)
        ])

        total_weight = sum([
            bit * item.weight
            for item, bit in zip(items, self.bits)
        ])

        if total_weight <= MAX_KNAPSACK_WEIGHT:
            return total_value
        
        return 0
        """
        total_value = 0
        for i in range(len(self.bits)):
            total_value += self.bits[i]*items[i].value
        return total_value
    
    def weight(self) -> float:
        total_weight = sum([
            bit * item.weight
            for item, bit in zip(items, self.bits)
        ])
        return total_weight
        


MAX_KNAPSACK_WEIGHT = 60
CROSSOVER_RATE = 0.53
MUTATION_RATE = 0.013
REPRODUCTION_RATE = 0.70

items = [
    Item("A", 7, 9),
    Item("B", 4, 3),
    Item("C", 11, 10),
    Item("D", 8, 15),
    Item("E", 6, 4),
    Item("F", 9, 5),
    Item("G", 4, 3),
    Item("H", 11, 10),
    Item("I", 14, 18),
    Item("J", 3, 7)
    
]


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
        if not (Individual(bits).weight() <= MAX_KNAPSACK_WEIGHT):
            continue

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
    if population[0].fitness() >= population[1].fitness():
        parents.append(population[0])
    else:
        parents.append(population[1])
    
    # tournament between third and fourth
    if population[2].fitness() >= population[3].fitness():
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
    population = sorted(population, key=lambda i: i.fitness(), reverse=True)
    index1 = population[0].fitness()
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
            else:
                children = parents

        if (len(children) == 2):
            if not ((children[0].weight() <= MAX_KNAPSACK_WEIGHT) and (children[1].weight() <= MAX_KNAPSACK_WEIGHT)):
                #print("Childrens rejetés",children[0].weight(), children[1].weight())
                continue
            if not ((children[0].fitness() + children[1].fitness()) >= (parents[0].fitness() + parents[1].fitness())):
                #print("Childrens rejetés",children[0].weight(), children[1].weight())
                continue
            #print("Childrens passés",children[0].weight(), children[1].weight())
            next_gen.extend(children)
                
    next_gen = sorted(next_gen, key=lambda i: i.fitness(), reverse=True)
    index2 = next_gen[0].fitness()
    
    if index2 > index1 :
        #print("gen ameliorée")
        return next_gen[:len(population)]
    else :
        #print("gen gardée")
        return population
    


def print_generation(population: List[Individual]):
    for individual in population:
        print(individual.bits, individual.weight(), individual.fitness())
    print()
    print("Average fitness", sum([x.fitness() for x in population])/len(population))
    print("-" * 32)


def average_fitness(population: List[Individual]) -> float:
    return sum([i.fitness() for i in population]) / len(population)


def solve_knapsack() -> Individual:
    population = generate_initial_population()

    avg_fitnesses = []

    for _ in range(200):
        avg_fitnesses.append(average_fitness(population))
        population = next_generation(population)

    population = sorted(population, key=lambda i: i.fitness(), reverse=True)
    return population[0]


if __name__ == '__main__':
    
    tab1 = [p for p in range(0, 50)]
    tab2 = [p for p in range(0, 50)]
    for i in range(50):
        solution = solve_knapsack()
        tab1[i] = solution.fitness()
        print(solution, solution.fitness())
        
        
    plt.plot(tab2, tab1)
    plt.show() # affiche la figure à l'écran
