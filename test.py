import numpy as np

arr = np.array([1,2,3,4,5,6])
arr = np.roll(arr, 1)

print(arr)

arr = np.zeros(5)
print(arr)


gameMemory = np.empty(shape=(0, 2), dtype=[('observation', object), ('action', int)])

print(gameMemory)
gameMemory = np.append(gameMemory, [(np.arange(4), 1)], axis=0)
print(gameMemory)
gameMemory = np.append(gameMemory, [(np.arange(4), 0)], axis=0)
print(gameMemory)

print(len(gameMemory[0][0]))


# arr = gameMemory['action']
# print(arr)
