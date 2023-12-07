First, you import numpy, initialize a random seed, so that you can reproduce the same results when running the program again, and import the keras objects you’ll 
use to build the neural network.

```python
import numpy as np
np.random.seed(444)
```

Then, you define an X array, containing the 4 possible A-B sets of inputs for the XOR operation and a y array, containing the outputs for each of the sets of inputs defined in X.

```python
X = np.array([[0, 0], 
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])
```

The next five lines define the neural network. The Sequential() model is one of the models provided by keras to define a neural network, in which the layers of the network are defined in a sequential way. 
Then you define the first layer of neurons, composed of two neurons, fed by two inputs, defining their activation function as a sigmoid function in the sequence. Finally, you define the output layer composed of one neuron with the same activation function.

```python
model = Sequential()
model.add(Dense(2, input_dim=2))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
```


The following two lines define the details about the training of the network. To adjust the weights of the network, you’ll use the Stochastic Gradient Descent (SGD) with the learning rate equal to 0.1, and you’ll use the mean squared error as a loss function to be minimized.

```python
sgd = SGD(lr=0.1)
model.compile(loss='mean_squared_error', optimizer=sgd)
```

Finally, you perform the training by running the fit() method, using X and y as training examples and updating the weights after every training example is fed into the network (batch_size=1). The number of epochs represents the number of times the whole training set will be used to train the neural network.

```python
model.fit(X, y, batch_size=1, epochs=5000)
```

In this case, you’re repeating the training 5000 times using a training set containing 4 input-output examples. By default, each time the training set is used, the training examples are shuffled.

On the last line, after the training process has finished, you print the predicted values for the 4 possible input examples.

```python
if __name__ == '__main__':
    print(model.predict(X))
```

By running this script, you’ll see the evolution of the training process and the performance improvement as new training examples are fed into the network.
After the training finishes, you can check the predictions the network gives for the possible input values:
![XOR Result](../../images/nnxorResult.png)

As you defined X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), the expected output values are 0, 1, 1, and 0, which is consistent with the predicted outputs of the network, given you should round them to obtain binary values.