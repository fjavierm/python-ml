import matplotlib.pyplot as plt

sleep = [7, 8, 6, 11, 7]
eat = [2, 3, 4, 3, 2]
work = [7, 8, 7, 2, 2]
fun = [8, 5, 7, 8, 13]

splitters = [7, 2, 2, 13]

activities = ['Sleep', 'Eat', 'Work', 'Fun']
colors = ['red', 'purple', 'blue', 'orange']

plt.pie(splitters,
        labels=activities,
        colors=colors,
        startangle=90,
        shadow=True,
        explode=(0.1, 0, 0, 0),
        autopct='%1.1f%%')

plt.title('Pie charts')

plt.show()

quit()
