import matplotlib.pyplot as plt
from genetic import solve_knapsack, print_generation

if __name__ == '__main__':
    y = []
    x = [x for x in range(100)]
    solutions = []
    for _ in range(100):
        solution = solve_knapsack()
        y.append(solution.fitness())
        solutions.append(solution)

    print_generation(solutions, True, False)
    # plt.figure()
    # plt.plot(x,y)
    # plt.show()
    