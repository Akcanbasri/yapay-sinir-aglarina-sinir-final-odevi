# Iris Classification using Neural Network

This repo contains a Python code that uses a neural network model to classify the iris dataset into three species: setosa, versicolor, and virginica. The code uses the pandas, numpy, matplotlib, seaborn, sklearn, and keras modules to perform data preprocessing, model building, training, and evaluation.

## Dataset

The dataset is called 'iris.csv' and it contains 150 rows and 5 columns. The columns are sepal length, sepal width, petal length, petal width, and species. The dataset is used to classify the iris flowers based on their features.

## Data Preprocessing

The output variable (species) is encoded using label encoding and one-hot encoding to convert it into a numerical format. The input variables are scaled using standard scaler to have zero mean and unit variance.

## Data Splitting

The dataset is split into 70% training and 30% testing using the train_test_split function from the sklearn module. The random_state parameter is set to 42 for reproducibility.

## Neural Network Model

The neural network model is created using the sequential class from the keras module. It has two dense layers: the first one has 8 units and ReLU activation function, and the second one has 3 units and softmax activation function. The input dimension of the first layer is based on the number of input features (4), and the output dimension of the second layer is based on the number of output classes (3).

## Model Compilation

The model is compiled with the stochastic gradient descent (SGD) optimizer, categorical crossentropy loss function, and accuracy metric.

## Model Training

The model is fitted to the training data with a batch size of 16 and 70 epochs.

## Model Evaluation

The model is evaluated on the testing data and the accuracy and loss are printed.

## Code

The code is given below:

```python
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn import preprocessing  
from sklearn.model_selection import train_test_split 
from keras.models import Sequential  
from keras.layers import Dense  
iris = pd.read_csv('iris.csv')  # Reading the 'iris.csv' file and storing the data in the 'iris' DataFrame
df = iris.copy()  # Creating a copy of the 'iris' DataFrame and assigning it to 'df' for further operations
df.head()  # Displaying the first few rows of the DataFrame 'df'
print(df.shape)  # Printing the shape of the DataFrame 'df', which shows the number of rows and columns
print(df.info())  # Printing the information about the DataFrame 'df', including column names, data types, and memory usage
df.describe().T # Displaying the descriptive statistics of the DataFrame 'df'
# Splitting the input and output variables
x = df.iloc[:, 0:4]  # Selecting all rows and the first 4 columns as the input variables and assigning it to 'x'
y = df.iloc[:, 4:5]  # Selecting all rows and the 5th column as the output variable and assigning it to 'y'
x.head() # Displaying the first few rows of the input variables
y.head() # Displaying the first few rows of the output variable
encoded_y = y.apply(preprocessing.LabelEncoder().fit_transform)  # Applying label encoding to the output variable 'y' using the LabelEncoder from the preprocessing module
ohenc = preprocessing.OneHotEncoder()  # Creating an instance of the OneHotEncoder from the preprocessing module
y = ohenc.fit_transform(encoded_y).toarray()  # Applying one-hot encoding to the encoded output variable 'encoded_y' and converting it to a NumPy array
# Creating a new DataFrame 'y' with categorical labels
# setosa = [1, 0, 0], versicolor = [0, 1, 0], virginica = [0, 0, 1]
y = pd.DataFrame(y, columns=['setosa', 'versicolor', 'virginica'])
# Using StandardScaler for feature scaling
scaler = preprocessing.StandardScaler()  # Creating an instance of the StandardScaler from the preprocessing module
x = scaler.fit_transform(x)  # Scaling the input variables 'x' using the fit_transform() method of the scaler
# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
enrty_number = x_train.shape[1]  # Storing the number of input features in the variable 'enrty_number'
class_number = y_train.shape[1]  # Storing the number of output classes in the variable 'class_number'
model = Sequential()  # Creating a Sequential model
model.add(Dense(8, input_dim=enrty_number, activation='relu'))  # Adding a dense layer with 8 units, input dimension based on 'enrty_number', and ReLU activation function
model.add(Dense(class_number, activation='softmax'))  # Adding a dense layer with 'class_number' units and softmax activation function
model.summary()  # Displaying the summary of the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])  # Compiling the model with the specified optimizer, loss function, and metrics
model.fit(x_train, y_train, batch_size=16, epochs=70)  # Fitting the model to the training data with the specified batch size and number of epochs
result = model.evaluate(x_test, y_test, verbose=0)  # Evaluating the model on the testing data and storing the results in the 'result' variable
print("Accuracy: %.2f%%" % (result[1] * 100))
print("Loss: %.2f%%" % (result[0] * 100))
```
