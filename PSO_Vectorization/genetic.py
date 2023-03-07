from typing import List
import numpy as np
import pandas as pd
import random as rd
from random import randint
import matplotlib.pyplot as plt

from decorator import bench
from PSOKnapsak import fitness_function


item_number = np.arange(1,11)
num_generations = 25


def cal_fitness(weight, value, population, threshold):
    fitness = np.empty(population.shape[0])
    for i in range(population.shape[0]):
        S1 = np.sum(population[i] * value)
        S2 = np.sum(population[i] * weight)
        if S2 <= threshold:
            fitness[i] = S1
        else :
            fitness[i] = 0 
    return fitness.astype(int), S2


def selection(fitness, num_parents, population):
    fitness = list(fitness)
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        parents[i,:] = population[max_fitness_idx[0][0], :]
        fitness[max_fitness_idx[0][0]] = -999999
    return parents


def crossover(parents, num_offsprings):
    offsprings = np.empty((num_offsprings, parents.shape[1]))
    crossover_point = int(parents.shape[1]/2)
    crossover_rate = 0.8
    i = 0
    
    while (parents.shape[0] < num_offsprings):
        parent1_index = i%parents.shape[0]
        parent2_index = (i+1)%parents.shape[0]
        x = rd.random()
        if x > crossover_rate:
            continue
        parent1_index = i%parents.shape[0]
        parent2_index = (i+1)%parents.shape[0]
        offsprings[i,0:crossover_point] = parents[parent1_index,0:crossover_point]
        offsprings[i,crossover_point:] = parents[parent2_index,crossover_point:]
        i =+ 1
    
    return offsprings 


def mutation(offsprings):
    mutants = np.empty((offsprings.shape))
    mutation_rate = 0.4
    for i in range(mutants.shape[0]):
        random_value = rd.random()
        mutants[i,:] = offsprings[i,:]
        if random_value > mutation_rate:
            continue
        int_random_value = randint(0,offsprings.shape[1]-1)    
        if mutants[i,int_random_value] == 0 :
            mutants[i,int_random_value] = 1
        else:
            mutants[i,int_random_value] = 0
    return mutants


def optimize(weight, value, population, pop_size, num_generations, threshold):
    parameters, fitness_history, weight_history = [], [], []
    num_parents = int(pop_size[0]/2)
    num_offsprings = pop_size[0] - num_parents 
    for i in range(num_generations):
        fitness, population_weight = cal_fitness(weight, value, population, threshold)
        fitness_history.append(fitness)
        weight_history.append(population_weight)
        parents = selection(fitness, num_parents, population)
        offsprings = crossover(parents, num_offsprings)
        mutants = mutation(offsprings)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants
        
    fitness_last_gen, weight_last_gen = cal_fitness(weight, value, population, threshold)      
    best_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
    parameters.append(population[best_fitness[0][0],:])
    return parameters, best_fitness, fitness_history, weight_history


@bench
def solve_knapsack(W: int, wt: List[int], val: List[int]):
    item_number = np.arange(1, wt.shape[0]+1)
    solutions_per_pop = 8
    pop_size = (solutions_per_pop, item_number.shape[0])
    initial_population = np.random.randint(2, size = pop_size)
    initial_population = initial_population.astype(int)
    
    parameters, best_fitness, fitness_history, weight_history = optimize(wt, val, initial_population, pop_size, num_generations, W)
    
    selected_items = item_number * parameters
    
    fitness_history_mean = [np.mean(fitness) for fitness in fitness_history]
    fitness_history_max = [np.max(fitness) for fitness in fitness_history]
    fitness_history_std = [np.std(fitness) for fitness in fitness_history]
    
    fitness_value = fitness_function(parameters,wt,val,W)

    return parameters,fitness_value

