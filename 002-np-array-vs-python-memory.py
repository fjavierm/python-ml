import sys
import numpy as np

s = range(1000)
print('Python list: ')
print(sys.getsizeof(5) * len(s))
print()

d = np.arange(1000)
print('NumPy array: ')
print(d.size * d.itemsize)

quit()