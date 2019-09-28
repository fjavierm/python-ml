# Machine Learning with Python
Notes about learning Machine Learning with Python. The purpose is to learn libraries and tools related with Python in the scope of Machine Learning.

## Available languages
Obviously, we can write Machine Learning code in multiple languages  but, there are four languages that, in the scope of Machine Learning, are taking the head of the race.

* (Python)[https://www.python.org]
  * It is the leader of the race right now due to the simplicity and its soft learning curve.
  * It is specially good and successful for beginners, in both, programming and Machine Learning.
  * The libraries ecosystem and the community support is huge.
  
* (R)[https://www.r-project.org]
  * It is design for statistical analysis and visualization, it is used frequently to unlock patterns in big data blocks.
  * With RStudio, developers can easily build algorithms and statistical visualization.
  * It is a free alternative to more expensive software like Matlab.
  
* (Matlab)[https://www.mathworks.com/products/matlab.html]
  * It is fast, stable and secure for complex mathematics.
  * It is considered as a hardcore language for mathematicians and scientists.
  
* (Julia)[https://julialang.org]
  * Designed to deal with numerical analysis needs and computational science.
  * The base Julia library was integrated with C and Fortram open source libraries.
  * The collaboration between the Jupyter and Julia communities, it gives Julia a powerful UI.
  
Some important points to consider when choosing a language should be:

* Speed.
* Learning curve.
* Cost.
* Community support.
* Productivity.

Here we can classify our languages as follows:

* Speed: R is basically an statistical language and it is difficult to beat in this context.
* Learning curve: Here depends on the person's knowledge. R is closer to the functional languages as opposite to python that is closer to object oriented languages.
* Cost: Only Matlab is not a free language. The other languages are open source.
* Community: All of them are very popular but, Python has the bigger community and amount of resources available.
* Productivity: R for statistical analysis, Matlab for computational vision, bio-informatics or biology is the playground of Julia and, Python is the king for general tasks and multiple usages.

The decision, at the end of the day, it is about a balance between all the characteristics seen above, our skills and the field we are or the tasks we want to implement.

In my case, I am going to choose Python as probably all of you have assumed because it is like a swiss knife and at this point, the beginning, I think this is important. There is always time later to focus on other things or reduce the scope.

## IDEs

There are multiple IDEs that support Python. As a very extended language there are multiple tools and environments we can use. Here just take the one you like the more.

If you do not know any IDE or platform, there are two of them that a lot of Data Scientist use:

* (Jypiter Notebook)[https://jupyter.org].
* (Spyder)[https://www.spyder-ide.org].

I do not know them, as a developer I am more familiar with Visual Studio Code or IntelliJ, and I will be using one of them probably unless I discover some exciting functionality or advantage in one of the other.

## Python library - (NumPy)[https://numpy.org]

As I have said before, one of the best advantages of Python is the huge community and amount of resource that supports it. On of the libraries we can find it is NumPy (NUMerical PYthon).

It is one of the main libraries to support scientific work with Python. It brings powerful data structures and implements matrices and multidimensional matrices.

#### Example

[001-numpy-example](./001-numpy-example.py)

```python
import numpy as np

a = np.array([1, 2, 3])
print('1D array:')
print(a)
print()

b = np.array([(1, 2, 3), (4, 5, 6)])
print('2D array:')
print(b)
```

### NumPy structures vs Python structures

Obviously the question 'why should I use the NumPy structures and not the Python available structures?' should pop up quickly in our minds. There are a few reasons for that:

* NumPy arrays consumes less memory than Python lists.
* They are faster in execution terms.

#### Example

[002-np-array-vs-python-memory](./002-np-array-vs-python-memory.py)

```python
import sys
import numpy as np

s = range(1000)
print('Python list: ')
print(sys.getsizeof(5) * len(s))
print()

d = np.arange(1000)
print('NumPy array: ')
print(d.size * d.itemsize)
```

#### Example

[003-np-array-vs-python-speed](./003-np-array-vs-python-speed.py)

```python
import time
import numpy as np

SIZE = 1_000_000

L1 = range(SIZE)
L2 = range(SIZE)
A1 = np.arange(SIZE)
A2 = np.arange(SIZE)

start = time.time()
result = [(x, y) for x, y in zip(L1, L2)]
print('Python list: ')
print((time.time() - start) * 1000)
print()

start = time.time()
result = A1 + A2
print('NumPy array: ')
print((time.time() - start) * 1000)
```

In addition to the speed improvement, it is worth it to point to the difference of how Python and NumPy perform the operation to calculate the result:

* `[(x, y) for x, y in zip(L1, L2)]`
* A1 + A2

As you can see, NumPy is much easier to write and to understand.

### NumPy useful methods

* Creating matrices:
  * `import numpy as np`- Import the NumPy dependency.
  * `np.array()` - Creates a matrix.
  * `np.ones((3, 4))` - Creates a matrix with a one in every position.
  * `np.zeros((3, 4))` - Creates a matrix with a zero in every position.
  * `np.random.random((3, 4))` - Creates a matrix with random values in every position.
  * `np.empty((3, 4))` - Creates an empty matrix.
  * `np.full((3, 4), 8)` - Creates a matrix with a specified value in every position.
  * `np.arange(0, 30, 5)` - Creates a matrix with a distribution of values (from 0 to 30 every 5).
  * `np.linspace(0, 2, 5)` - Creates a matrix with a distribution of values (5 elements from 0 to 2).
  * `np.eye(4, 4)`- Creates an identity matrix.
  * `np.identity(4)`- Creates an identity matrix.

* Inspect matrices (_assuming `a = np.array([(1, 2, 3), (4, 5, 6)])` and `b = ...`_)
  * `a.ndim`- Matrix dimension.
  * `a.dtype` - Matrix data type.
  * `a.size` - Matrix size.
  * `a.shape` - Matrix shape.
  * `a.reshape(3, 2)` - Change the shape of a matrix.
  * `a[3, 2]` - Select a single element of the matrix.
  * `a[0:, 2]` - Extract the value in the column 2 from every row.
  * `a.min()`, `a.max()` and `a.sum()`- Basic operations over the matrix.
  * `np.sqrt(a)` - Square root of the matrix.
  * `np.std(a)` - Standard deviation of the matrix.
  * `a + b`, `a - b`, `a * b` and `a / b` - Basic operations between matrices.
  
## Python library - [pandas](https://pandas.pydata.org)

pandas (PANel DAta) library is another library available for Python environments. It is a very popular library too. It is an open source library and provide high performance analysis and manipulation data tools.

This library can help us to execute five common steps in data analysis:

* Load data.
* Data preparation.
* Data manipulation.
* Data modeling.
* Data analysis.

The main panda structure is the `DataFrame`. Two-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes. It is composed by three elements: the data, the index and the columns. In addition, the names of columns and indexes cn be specified.

### panda vs NumPy

The main difference is how objects look like

```
NumPy 1D array         pandas - Series
                       index ->  1  2  3
[1, 2, 3]                       [4, 5, 6]

NumPy 2D array         pandas - DataFrame
                       index ->  1  2  3
[1, 2, 3]                     A  [4, 5, 6]
[4, 5 ,6]                     B  [7, 8, 9]
```

### Main library characteristics

* The DataFrame object is fast and efficient.
* Tools to load data in memory from different formats.
* Data alignment and missing data management.
* Remodeling and turning date sets
* Labeling, cut and indexation of big amounts of data.
* Columns can be removed or inserted.
* Data grouping for aggregation and transformation.
* High performance for data union and merge.
* Time based series functionality.
* Has three main structures:
  * Series: 1D structures.
  * DataFrame: 2D structures.
  * Panel: 3D structures.

### Installing pandas

pandas library is not present in the default Python installation and needs to be installed:

`pip install pandas`

To use it, you just need to import it like any other Python library:

`import pandas as pd`

### pandas useful methods

#### Creating a Series

````python
import pandas as pd

series = pd.Series({"UK": "London",
                    "Germany": "Berlin",
                    "France": "Paris",
                    "Spain": "Madrid"})
print(series)
````

#### Creating a DataFrame

[004-pd-dataframe](./004-pd-dataframe.py)

```python
import numpy as np
import pandas as pd

data = np.array([['', 'Col1', 'Col2'], ['Fila1', 11, 22], ['Fila2', 33, 44]])
print(pd.DataFrame(data=data[1:, 1:],
                   index=data[1:, 0],
                   columns=data[0, 1:]))
```

Without boilerplate code:

````python
import numpy as np
import pandas as pd

df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
print(df)
````

#### Exploring a DataFrame

* `df.shape` - DataFrame shape.
* `len(df.index)`- DataFrame high.
* `df.describe()` - DataFrame numeric statistics (_count, mean, std, min, 25%, 50%, 75%, max_).
* `df.mean()` - Return the mean of the values for the requested axis.
* `df.corr()` - Correlation of columns.
* `df.count()` - Count of non-null values per column.
* `df.max()` - Maximum value per column.
* `fd.min()` - Minimum per column.
* `df.median()` - Median value per column.
* `df.std()` - Standard deviation per column.
* `df[0]` - Select a DataFrame column (returned as a new DataFrame).
* `df[1, 2]` - Select two DataFrame columns (returned as a new DataFrame).
* `df.iloc[0][2]` - Select a value.
* `df.loc([0]` - Select a column using the index.
* `df.iloc([0, :]` - Select a column using the index.
* `pd.read_<file_type>()` - Read from a file (`pd.read_csv('train.csv')`.
* `df.to_<file_type>()` - Write to a file (`pd.to_csv('new_train.csv')`).
* `df.isnull()` - Verify is there are null values in the data set.
* `df.isnull().sum()` - Return the sum of null values per column in the data set.
* `df.dropna()` or `df.dropna(axis = 1)` - Remove rows or columns with missing data.
* `df.fillna(x)` - Replace missing values with `x` (`df.fillna(df.mean())`).

## Python library - [Matplotlib](https://matplotlib.org)

Matplotlib library is another library available for Python environments. It is a 