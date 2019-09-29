import matplotlib.pyplot as plt

a = [22, 55, 62, 45, 21, 22, 32, 42, 42, 4,
     2, 102, 95, 85, 55, 110, 120, 70, 65, 55,
     111, 115, 80, 75, 65, 54, 44, 50, 60, 70,
     80, 90, 100]
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

plt.hist(a, bins, histtype='bar', rwidth=0.8, color='lightgreen')

plt.title('Histograms')
plt.xlabel('X Axi')
plt.ylabel('Y Axi')

plt.legend()
plt.grid()

plt.show()

quit()
