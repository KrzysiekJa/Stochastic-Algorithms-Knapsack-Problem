import matplotlib.pyplot as plt
import numpy as np
from genetic2 import solve_knapsack

if __name__ == '__main__':
    ## KNAPSACKs parameters
    W = 60
    wt = [7,4,11,8,6,9,4,11,14,3]
    val = [9,3,10,15,4,5,3,10,18,7]
    n = 10

    y = []
    weights = []
    solutions = []
    avg_means = [] # Moyenne des moyennes des fitnesses
    best_fitnesses = [] # meilleurs fitnesses
    elapsed_times = [] # Array of elapsed times
    
    time, (solution, best_fitness, fitness_hist_mean,fitness_hist_max, fitness_hist_std,fitness_hist, weight_hist) = solve_knapsack(W, wt, val)
    print(f"Elapsed time: {time:.5f} seconds")
    y.append(best_fitness)
    #weights.append(solution.weight())
    solutions.append(solution)
    last_best_fitness = best_fitness
    print("last_best_fitness : ", last_best_fitness)
    
    
    
    for _ in range(10):
        time, (solution, best_fitness, fitness_hist_mean,fitness_hist_max, fitness_hist_std,fitness_hist, weight_hist) = solve_knapsack(W, wt, val)
        elapsed_times.append(time)
        best_fitnesses.append(max(best_fitness))
        avg_means.append(np.mean(fitness_hist_mean))
        if last_best_fitness < best_fitness:
            y.append(best_fitness)
            #weights.append(solution.weight())
            solutions.append(solution)
            last_best_fitness = best_fitness

    # print(f'Best fitnesses: {best_fitnesses}')
    # print(f'avg weights: {avg_means}')
    print(f"Elapsed time: {sum(elapsed_times):.5f} seconds")
    x = np.arange(len(solutions))
    #print_generation(solutions)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10,6))
    rects1 = ax1.bar(x - 0.35/2,y, 0.35, label="Value")
    rects2 = ax1.bar(x + 0.35/2,weights, 0.35, label="Weight")
    ax1.legend()
    ax1.set_xticks(x, [x for x in range(1,len(solutions)+1)])
    ax1.bar_label(rects1, padding=3)
    ax1.bar_label(rects2, padding=3)
    ax1.set_title("Candidates validated")

    # On affiche la moyenne des moyennes des fitness ainsi que les meilleures fitnesses de chaque itération
    ax2.plot([x for x in range(len(avg_means))], avg_means, label="Average fitness mean")
    ax2.plot([x for x in range(len(best_fitnesses))], best_fitnesses, label="Best fitnesses")
    ax2.legend()
    ax2.set_title("Average weights each iteration")

    fig.tight_layout()
    plt.show()
    
    """
    y = []
    weights = []
    x = [x for x in range(200)]
    solutions = []
    for _ in range(200):
        solution = solve_knapsack()
        y.append(solution.fitness())
        weights.append(solution.weight())
        solutions.append(solution)

    print_generation(solutions)
    plt.figure()
    plt.plot(x,y)
    plt.plot(x, weights)
    plt.show()
    """