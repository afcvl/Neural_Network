import random

class Perceptron():
    def __init__(self):
        self.weights = []
    
    def process_input(self, inputs):
        sum = 0
        for i in range(len(self.weights)):
            sum =+ inputs[i]*self.weights[i]
            
        return self.ReLU(sum)
    
    def ReLU(self, value):
        if value > 0:
            return value
        else:
            return 0
    
    def init_weights(self, n_weights):
        for i in range(n_weights):
            self.weights.append(random.uniform(-1,1))
    
class MLP():
    def __init__(self, n_inputs, n_exits, n_hiden_layer):
        self.inputs = []
        self.exits = []
        self.hidden = []
        
        for i in range(n_inputs):
            self.inputs.append(Perceptron())
            
        for i in range(n_hiden_layer):
            self.hidden.append(Perceptron())
            self.hidden[i].init_weights(n_inputs)
        
        for i in range(n_exits):
            self.exits.append(Perceptron())
            self.exits[i].init_weights(n_hiden_layer)
            
    def foward(self):
        pass
            
            
neuron = Perceptron()

values = [5,3,4,2,8,6]

neuron.init_weights(len(values))

print(neuron.process_input(values))