import matplotlib.pyplot as plt
import numpy as np
from genetic import solve_knapsack, print_generation, Individual

if __name__ == '__main__':
    y = []
    weights = []
    solutions = []
    
    solution = solve_knapsack()
    y.append(solution.fitness())
    weights.append(solution.weight())
    solutions.append(solution)
    
    for _ in range(500):
        solution = solve_knapsack()
        if solutions[-1].fitness() < solution.fitness():
            y.append(solution.fitness())
            weights.append(solution.weight())
            solutions.append(solution)
        
    x = np.arange(len(solutions))
    #print_generation(solutions)
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 0.35/2,y, 0.35, label="Value")
    rects2 = ax.bar(x + 0.35/2,weights, 0.35, label="Weight")
    ax.legend()
    ax.set_xticks(x, [x for x in range(1,len(solutions)+1)])
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
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
    