import numpy as np
from keras.layers import Dense, Input
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
# define the model
from keras.models import Sequential   

#load data
data = np.loadtxt('concrete_data.csv', delimiter=',', skiprows=1)
n_cols = data.shape[1]  # number of input features
X = data[:, 0:n_cols-1]  # input features
y = data[:, n_cols-1]  # target feature

#preprocess data
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y.reshape(-1, 1) 
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)
 





# define the model
def baseline_model():
    n_cols = 8  # Set this to match the number of features in your input data
    model = Sequential()
    model.add(Input(shape=(n_cols,)))  # Input layer
    model.add(Dense(10, activation='relu'))  # Hidden layer
    model.add(Dense(1))  # Output layer
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    return model

mse_scores = []  # list to store all mean squared errors
# Repeat steps 1-3 for 50 times
for i in range(50):
    print(f'cycle:', i, 'of 50')
    model = baseline_model()  # create your model
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # Split the data

    model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)  # Fit the model

    scores = model.evaluate(X_test, y_test, verbose=1)  # evaluate the model with mean squared error
    print(f'Mean Squared Error: {scores[1]}')
    mse_scores.append(scores)  
    # append the score to the list

# Report the mean and the standard deviation of the mean squared errors
mse_scores = np.array(mse_scores)  # convert list to numpy array for calculations
print("Mean of Mean Squared Errors: %.2f" % mse_scores.mean())
print("Standard Deviation of Mean Squared Errors: %.2f" % mse_scores.std())
