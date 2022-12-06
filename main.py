import matplotlib.pyplot as plt
from genetic import solve_knapsack, print_generation

if __name__ == '__main__':
    y = []
    weights = []
    x = [x for x in range(100)]
    solutions = []
    for _ in range(100):
        solution = solve_knapsack()
        y.append(solution.fitness())
        weights.append(solution.weight())
        solutions.append(solution)

    print_generation(solutions)
    plt.figure()
    plt.plot(x,y)
    plt.plot(x, weights)
    plt.show()
    