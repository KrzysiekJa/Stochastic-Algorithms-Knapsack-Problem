import random
from typing import List
import numpy as np
import matplotlib.pyplot as plt

from decorator import bench


class Item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value


class Individual:
    def __init__(self, bits: List[int], wt: List[int], val: List[int]):
        self.bits: List[int] = bits
        self.items: List[Item] = [Item(e[0],e[1]) for e in zip(wt, val)]

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
            total_value += self.bits[i]*self.items[i].value
        return total_value
    
    def weight(self) -> float:
        total_weight = sum([
            bit * item.weight
            for item, bit in zip(self.items, self.bits)
        ])
        return total_weight
        


# MAX_KNAPSACK_WEIGHT = 60
CROSSOVER_RATE = 0.53
MUTATION_RATE = 0.013
REPRODUCTION_RATE = 0.70

# items = [
#     Item("A", 7, 9),
#     Item("B", 4, 3),
#     Item("C", 11, 10),
#     Item("D", 8, 15),
#     Item("E", 6, 4),
#     Item("F", 9, 5),
#     Item("G", 4, 3),
#     Item("H", 11, 10),
#     Item("I", 14, 18),
#     Item("J", 3, 7)
    
# ]


def generate_initial_population(W, wt, val, n, count=6) -> List[Individual]:
    population = set()

    # generate initial population having `count` individuals
    while len(population) != count:
        # pick random bits one for each item and 
        # create an individual 
        bits = [
            random.choice([0, 1])
            for _ in range(n)
        ]
        if not (Individual(bits, wt, val).weight() <= W):
            continue

        population.add(Individual(bits, wt, val))


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


def crossover(parents: List[Individual], wt: List[int], val: List[int], n: int) -> List[Individual]:
    N = n

    child1 = parents[0].bits[:N//2] + parents[1].bits[N//2:]
    child2 = parents[0].bits[N//2:] + parents[1].bits[:N//2]

    return [Individual(child1, wt, val), Individual(child2, wt, val)]


def mutate(individuals: List[Individual]) -> List[Individual]:
    for individual in individuals:
        for i in range(len(individual.bits)):
            if random.random() < MUTATION_RATE:
                # Flip the bit
                individual.bits[i] = ~individual.bits[i]


def next_generation(population: List[Individual], MAX_KNAPSACK_WEIGHT: int, wt: List[int], val: List[int], n: int) -> List[Individual]:
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
                children = crossover(parents, wt, val, n)
            
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


# W: int, max weight
# wt: int[], weights
# val: int[], values
# n: int, number of items
@bench
def solve_knapsack(W: int, wt: List[int], val: List[int], n: int) -> tuple[Individual, List[float], List[float]]:
    population = generate_initial_population(W, wt, val, n)

    avg_fitnesses = []
    best_fitnesses = []

    for _ in range(200):
        avg_fitnesses.append(average_fitness(population))
        population = sorted(population, key=lambda i: i.fitness(), reverse=True)
        best_fitnesses.append(population[0].fitness())
        population = next_generation(population, W, wt, val, n)


    population = sorted(population, key=lambda i: i.fitness(), reverse=True)
    return (population[0], avg_fitnesses, best_fitnesses)


if __name__ == '__main__':
    
    #for i in range(1):

    time, (solution, avg, best_fitness) = solve_knapsack(60, [7,4,11,8,6,9,4,11,14,3], [9,3,10,15,4,5,3,10,18,7], 10)
    print(f"Elapsed time: {time:.5f} seconds")
    # tab1[i] = solution.fitness()
    # print(solution, solution.fitness())
    # print(avg)
    # print(best_fitness)
    x = [x for x in range(len(avg))]
    plt.figure()
    plt.plot(x, avg)
    plt.plot(x, best_fitness)
    plt.show() # affiche la figure à l'écran