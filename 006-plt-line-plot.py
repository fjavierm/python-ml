import matplotlib.pyplot as plt

x1 = [3, 4, 5, 6]
y1 = [5, 6, 3, 4]
x2 = [2, 5, 8]
y2 = [3, 4, 3]

plt.plot(x1, y1, label='Line 1', linewidth=4, color='blue')
plt.plot(x2, y2, label='Line 2', linewidth=4, color='green')

plt.title('Line Plot')
plt.xlabel('X Axi')
plt.ylabel('Y Axi')

plt.legend()
plt.grid()

plt.show()

quit()
