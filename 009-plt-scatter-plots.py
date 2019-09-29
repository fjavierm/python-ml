import matplotlib.pyplot as plt

x1 = [0.25, 1.25, 2.25, 3.25, 4.25]
y1 = [10, 55, 80, 32, 40]
x2 = [0.75, 1.75, 2.75, 3.75, 4.75]
y2 = [42, 26, 10, 29, 66]

plt.scatter(x1, y1, label='Data 1', color='red')
plt.scatter(x2, y2, label='Data 2', color='purple')

plt.title('Scatter plots')
plt.xlabel('X Axi')
plt.ylabel('Y Axi')

plt.legend()
plt.grid()

plt.show()

quit()
