import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """
        sizes represent the length of each layer of the network
        sizes[0] is the input vector
        initializes the biases and weights randomly
        """
        self.num_layers = len(sizes)
        self.sizes = sizes

        # taking the first layer is the input layer
        # for each layer in the network, generate an nx1 vector where n is the length of the layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # for each layer in the network, initialize an nxm weight matrix
        # where n is size of the current layer
        # and m is the size of the previous layer
        self.weights = [np.random.randn(y,x) for y,x in zip(sizes[1:], sizes[:-1])]


    def SGD(self, training_data, epochs, mini_batch_size, eta):  
        """ 
        Train the network using stochastic gradient descent.
        training_data is a list of tuples (x,y) representing a the input and the desired output.
        We will do "epoch" passes, where every pass throgh the data, we will update out weights
        """
        n = len(training_data)

        #for every pass through
        for _ in range(epochs):
            #shuffle the training data
            random.shuffle(training_data)
            
            #split the data into mini batches
            #each mini batch is the substring from data[k:k+size] 
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            
            #update the weights for each mini batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)



    def update_mini_batch(self, mini_batch, epsilon):
        """ 
        given a mini batch of the data, estimate the gradient of the Cost function using the batch
            and update the weights with step size epsilon 
        """
        #size of the batch
        m = len(mini_batch)

        #directional derivatives of b and w
        #indicating how much each b and w should change
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            #average of the directional derivatives in the mini batch
            small_nabla_b , small_nabla_w = self.backprop(x,y)
            nabla_b = [b + db / m for b, db in zip(nabla_b, small_nabla_b)]
            nabla_w = [w + dw / m for w, dw in zip(nabla_w, small_nabla_w)]

        #update the weights and biases accordingly
        self.biases = [b - epsilon * db for b, db in zip(self.biases, nabla_b)]
        self.weights = [w - epsilon * dw for w, dw in zip(self.weights, nabla_w)]   


    def backprop(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []

        for w,b in zip(self.weights, self.biases):
            z =  np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        
        delta = self.cost_derivative(activations[-1],y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
        


    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives 
        partial C_x/partial a for the output activations."""
        return (output_activations-y)  

    def feed_forward(self,a):
        """ returns the output vector if a is the input to the neural network"""
        for w,b in zip(self.weights, self.biases) :
            z = np.dot(w, a)+ b
            a = self.sigmoid(z)
        return a
    

    def sigmoid(self, z):
        """sigmoid function"""
        return 1.0/(1.0+np.exp(-z))
    
    def sigmoid_prime(self, z):
        """derivative of the sigmoid function"""
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
