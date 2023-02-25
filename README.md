# MNIST-from-scratch

A custom neural network to classify hand written digits from the MNIST dataset

The VisNet class is a neural network implemented in Python using the NumPy library for numerical computations. Here's a summary of what each method does:

init(self, input_size, hidden_size, output_size): The constructor method initializes the VisNet object with three attributes: the size of the input layer, the size of the hidden layer, and the size of the output layer. It also initializes the weights and biases for the two layers using random values.

cross_entropy_loss(self, y_true, y_pred): This method computes the cross-entropy loss between the predicted labels y_pred and the true labels y_true.

softmax(self, x): This method computes the softmax function on the input array x. The softmax function normalizes the input array so that its elements sum up to one.

forward(self, x): This method performs a forward pass through the neural network with the input x. It computes the output of the network by applying a dot product between the input and the first set of weights, adding the first set of biases, applying a non-linear activation function (tanh), computing the output of the second layer by applying a dot product between the output of the first layer and the second set of weights, adding the second set of biases, and finally applying the softmax function.

backward(self, x, y_true, y_pred, a1): This method performs a backward pass through the neural network to compute the gradients of the loss with respect to the weights and biases. It uses the predicted labels y_pred, the true labels y_true, the input x, and the output of the first layer a1 as inputs to compute the gradients.

train(self, x_train, y_train, learning_rate=0.1, batch_size=128, num_epochs=10): This method trains the neural network on a given dataset of input images x_train and labels y_train. It uses batch gradient descent with a given learning rate and batch size, and trains the network for a given number of epochs. For each batch, it performs a forward pass, computes the loss, performs a backward pass, and updates the weights and biases.

evaluate(self, x_test, y_test): This method evaluates the performance of the neural network on a given test dataset of input images x_test and labels y_test. It computes the accuracy of the network by comparing the predicted labels with the true labels, and prints the test accuracy in percentage.
