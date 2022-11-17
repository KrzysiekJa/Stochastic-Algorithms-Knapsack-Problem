from typing import List

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

items = [
    Item("A", 7, 13),
    Item("B", 5, 14),
    Item("C", 9, 21),
    Item("D", 8, 9)
]

MAX_KNAPSACK_WEIGHT = 20
CROSSOVER_RATE = 0.53
MUTATION_RATE = 0.013
REPRODUCTION_RATE = 0.3