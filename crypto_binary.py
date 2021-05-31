import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cirq
import pennylane as qml

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

# ---------------------------- Classes --------------------------------- #
class NeuralNetwork:
    def __init__(self, xs_data, ys_data, sample_size=8):
        self.xs_data = xs_data
        self.ys_data = ys_data
        self.model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=sample_size, input_shape=[sample_size])
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.025) #usual : 0.05
        self.loss = tf.keras.losses.MeanSquaredError()
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def trainModel(self, epoch):
        self.history = self.model.fit(x=self.xs_data, y=self.ys_data, epochs=epoch, verbose=0)
        plt.plot(self.history.history['loss'])
        plt.title("Evolution of loss through epochs")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        #plt.show()

    def prediction(self, bits_table):
        print("\n     *-----Prediction of "+str(bits_table)+" -----*")
        result = self.model.predict([bits_table])
        accurate_value = encrypt(bits_table)
        print("Predicted value : "+str(result[0])+". The correct value is : "+str(accurate_value))

        array = list([])
        for table in result:
            for bit in table:
                value = int(round(bit))
                if(value<0):
                    value=0
                array+=[value]

        accuracy = 0.0
        for i in range(len(bits_table)):
            #print("Valeur array en "+str(i)+" : "+str(array[i])+". Valeur de la table initiale : "+str(bits_table[i])+".\n")
            if array[i]==accurate_value[i]:
                accuracy = accuracy + 1

        accuracy = (accuracy/len(bits_table))*100
        print("Accuracy : "+str(accuracy)+"%") #Rates à modifier pour avoir un apprentissage efficace

class QuantumNeuralNetwork(NeuralNetwork):
    def __init__(self, xs_data, ys_data):
        self.xs_data = xs_data
        self.ys_data = ys_data

        #self.layer_out = tf.keras.layers.Dense(units=8, input_shape=[8])
        self.model = tf.keras.Sequential([
        Quantum_Layer
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.025) #usual : 0.05
        self.model.compile(optimizer=self.optimizer, loss=tf.keras.losses.MeanSquaredError())

class Quantum(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Quantum, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name = 'kernel', shape = (input_shape[1], self.output_dim), initializer = 'normal', trainable = True)
        super(Quantum, self).build(input_shape)

    def call(self, input_data):
        data = quantumMapping(tf.convert_to_tensor(input_data))
        return K.dot(data, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim) # Implémentation : Quantum(units, input_shape=[size])


# --------------------- Tests custom loss function --------------------- #
def mse(y_true, y_pred, hidden):
    error = y_true-y_pred
    return K.mean(K.square(error)) + K.mean(hidden)

def test():
    X = np.random.uniform(0,1, (1000,10))
    y = np.random.uniform(0,1, 1000)

    inp = Input((10,))
    true = Input((1,))
    x1 = Dense(32, activation='relu')(inp)
    x2 = Dense(16, activation='relu')(x1)
    out = Dense(1)(x2)

    m = Model([inp,true], out)
    m.add_loss( mse( true, out, x1 ) )
    m.compile(loss=None, optimizer='adam')
    m.fit(x=[X, y], y=None, epochs=3)

    ## final fitted model to compute predictions
    final_m = Model(inp, out)

# ---------------------------------------------------------------------- #

def encrypt(bytes_array):
    key_array = [1, 0, 0, 1, 1, 1, 0, 0]
    encrypted_array = list([])
    for i in range(0,len(bytes_array),1):
        encrypted_array.append((bytes_array[i]+key_array[i])%2)
    return encrypted_array

#Not used
def quantumMapping(bytes_array):
    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(8) #len(bytes_array)
    sim = cirq.Simulator()

    for i in range(len(bytes_array)):
        if bytes_array[i] == 1:
            circuit.append([cirq.X(qubits[i])])
            #circuit.append([cirq.Z(qubits[i])])
        circuit.append([cirq.measure(qubits[i])])
    print(circuit)

    #Get state_vector after mapping
    results = sim.simulate(circuit)
    print(results)
    samples = cirq.sample_state_vector(results.state_vector(), indices=[0, 1, 2, 3, 4, 5, 6, 7])
    print(samples[0])
    return samples[0]

#Not used
def quantumLossFunction(y, y_predicted):
    return tf.reduce_sum(tf.square(tf.subtract(y, y_predicted)))

def setSampleArrays(sample_size):
    xs = None
    ys = None
    if sample_size == 2:
        xs = np.array([ encrypt([0, 0]), encrypt([0, 1]) ], dtype=int)
        ys = np.array([ [0, 0], [0, 1] ], dtype=int)
    if sample_size == 8:
        xs = np.array([ encrypt([0, 0, 0, 0, 0, 0, 0, 0]), encrypt([1, 1, 1, 1, 1, 1, 1, 1]), encrypt([1, 0, 0, 0, 1, 1, 1, 1]), encrypt([0, 1, 1, 0, 0, 1, 1, 0]), encrypt([0, 1, 1, 1, 0, 0, 1, 1])], dtype=int)
        ys = np.array( [ [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1, 1, 0], [0, 1, 1, 1, 0, 0, 1, 1] ], dtype=int)
    return xs, ys

def Classic_Main():
    xs, ys = setSampleArrays()
    network = NeuralNetwork(xs, ys)
    network.trainModel(100)

    bits_table = [1, 0, 1, 0, 0, 0, 0, 0]
    network.prediction(bits_table)
    return 0

def Quantum_Main():
    xs, ys = setSampleArrays()
    quantumNetwork = QuantumNeuralNetwork(xs, ys)
    quantumNetwork.trainModel(100)

    bits_table = [1, 0, 1, 0, 0, 0, 0, 0]
    quantumNetwork.prediction(bits_table)
    return 0


def Main():
    print("\n\n\n--------------------------- Quantum Machine Learning ---------------------------\n\n")
    loop_boolean = 1
    networks = list([])

    nb_networks = int(input("Choose how many neural networks you want to create for this simulation : "))
    sample_size = int(input("\n\nPlease enter the size of a batch's sample : "))
    xs, ys = setSampleArrays(sample_size)

    for i in range(0, nb_networks, 1):
        neural_network_type = int(input("\n\nChoose your type of neural network.\n   1. Classic Neural Network\n   2. Quantum Neural Network\n\nYour choice : "))
        if neural_network_type==1:
            networks.append(NeuralNetwork(xs, ys, sample_size))
        else:
            networks.append(QuantumNeuralNetwork(xs, ys, sample_size))

    while(loop_boolean):
        print("\n\n*-------------------- Training neural networks --------------------*\n\n")
        for network in networks:
            network.trainModel(100)

        user_choice = int(input("\n[*] Main menu : \n   [1]. Test the models on a specific entry.\n   [2]. Train the models (100 epochs).\n   [3]. Quit the simulation.\n\nYour choice : "))

        if user_choice == 1:
            user_input = input("\nEnter an array to try (format : 1,0,1,1,1,0,0...) : ")
            print(user_input)
            value_to_test = user_input.split(',')
            print(value_to_test)
            int_array = list([])
            for bit in value_to_test:
                int_array.append(int(bit))

            for network in networks:
                network.prediction(int_array)
            garbage_collector = input("\n----- Enter anything to continue. -----\n")

        if user_choice == 2:
            for network in networks:
                network.trainModel(100)

        if user_choice == 3:
            print("\n----- Exit -----\n")
            loop_boolean = 0


# -----------------------Quantum Layer initialization------------------- #
np.random.seed(1234)
n_qubits = 2

dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev)
def qnode(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
n_layers = 6
weight_shapes = {"weights": (n_layers, n_qubits)}
Quantum_Layer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)

Main()

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
