import matplotlib.pyplot as plt
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
        
    x = [x for x in range(len(solutions))]
    print_generation(solutions)
    plt.figure()
    plt.plot(x,y)
    plt.plot(x, weights)
    plt.show()
    