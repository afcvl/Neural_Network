import random
from Activations import Activation
        
class Perceptron():
    def __init__(self, actvation):
        self.weights = []
        self.activation = None
        
        match actvation:
            case 'relu':
                self.activation = Activation.ReLU
                
            case 'sigmoid':
                pass
            
            case 'tanh':
                pass
            
            case _:
                pass
            
    
    def process_input(self, inputs):
        sum = 0
        for i in range(len(self.weights)):
            sum =+ inputs[i]*self.weights[i]
            
        return self.activation(sum)
    
    def init_weights(self, n_weights):
        for i in range(n_weights):
            self.weights.append(random.uniform(-1,1))
    
class MLP():
    def __init__(self, n_inputs, n_hiden_layer, n_exits, activation='relu'):
        self.inputs = []
        self.exits = []
        self.hidden = []
        
        self.network = [self.inputs, self.hidden, self.exits]
        
        for i in range(n_inputs):
            self.inputs.append(Perceptron(activation))
            
        for i in range(n_hiden_layer):
            self.hidden.append(Perceptron(activation))
            self.hidden[i].init_weights(n_inputs)
        
        for i in range(n_exits):
            self.exits.append(Perceptron(activation))
            self.exits[i].init_weights(n_hiden_layer)
            
    def foward(self, input_data):
        output_layer1 = input_data
        output_layer2 = []
        output_layer3 = []
        
        for neuron in self.hidden:
            output_layer2.append(neuron.process_input(output_layer1))
            
        for neuron in self.exits:
            output_layer3.append(neuron.process_input(output_layer2))
            
        return output_layer3
            

values = [5,3,4,2,8,6]

nn = MLP(6,3,5)

print(nn.foward(values))