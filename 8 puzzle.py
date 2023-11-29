def water_jug(jug1, jug2, target):
    return [(a, b) for a in range(target + 1) for b in range(target + 1) if a + b == target and (a <= jug1 and b <= jug2)]
jug1_capacity = int(input("Enter the capacity of Jug 1: "))
jug2_capacity = int(input("Enter the capacity of Jug 2: "))
target_amount = int(input("Enter the target amount of water: "))
solutions = water_jug(jug1_capacity, jug2_capacity, target_amount)
print("Possible solutions:")
print(solutions)
