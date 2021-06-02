import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import cirq
import pennylane as qml

import numpy as np
import matplotlib.pyplot as plt
import random
import time

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


np.random.seed(1234)

"""
TO-DO LIST :

[x] Comprendre le fonctionnement de la bibliothèque pennylane ainsi que du layer quantique
implémenté. Etudier la possibilité de modifier sur-mesures le layer si nécessaire.
=> Les templates utilisés sont expliqués à l'adresse "https://pennylane.readthedocs.io/en/stable/introduction/templates.html"

[*] Modifier les réseaux de neurones pour minimiser la perte (le réseau quantique est particulièrement
inefficace pour un décryptage 8-bits => à voir /!\).
    * Simulation 2 bits :
        - Réseau classique :
            # learning_rate = 0.025 | 0.08 | 0.1
            # layers = Dense
        - Réseau quantique :
            # learning_rate = 0.05
            # layers = QuantumLayer

[*] Faire une démo optimisée de l'apprentissage du XOR 8-bits pour comparer théorie classique et quantique
+ chiffrer temporellement l'efficacité des solutions.

[*] Penser la phase 2 du projet : cryptage sur 16/32 bits ? Conversion textuelle directe ?
"""


# ---------------------------- Classes --------------------------------- #
class NeuralNetwork:
    def __init__(self, xs_data, ys_data, sample_size=8, number=None):
        self.name = None
        if number!=None:
            self.name = "Neural Network n°"+str(number)
        else:
            self.name = "Neural Network"
        self.xs_data = xs_data
        self.ys_data = ys_data
        self.model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=sample_size, input_shape=[sample_size])
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.08) #0.08 Tx = 75%
        self.loss = tf.keras.losses.MeanSquaredError()
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def trainModel(self, epoch):
        self.training_data_results = self.model.fit(x=self.xs_data, y=self.ys_data, epochs=epoch, verbose=0)

    def prediction(self, bits_table):
        print("\n\n*-----Prediction of "+str(bits_table)+" -----*")
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
    def __init__(self, xs_data, ys_data, number_of_qubits=8, number=None):
        self.name = None
        if number!=None:
            self.name = "Quantum Neural Network n°"+str(number)
        else:
            self.name = "Quantum Neural Network"
        self.xs_data = xs_data
        self.ys_data = ys_data

        #Quantum Layer creation
        dev = qml.device("default.qubit", wires=number_of_qubits)
        @qml.qnode(dev)
        def qnode(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(number_of_qubits))
            qml.templates.BasicEntanglerLayers(weights, wires=range(number_of_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(number_of_qubits)]
        n_layers = 6
        weight_shapes = {"weights": (n_layers, number_of_qubits)}
        Quantum_Layer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=number_of_qubits)


        self.model = tf.keras.Sequential([
        #tf.keras.layers.Dense(units=number_of_qubits, input_shape=[number_of_qubits]),
        Quantum_Layer,
        #tf.keras.layers.Dense(number_of_qubits, activation='relu')
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.065)
        self.model.compile(optimizer=self.optimizer, loss=tf.keras.losses.MeanSquaredError())

#Not used
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
def generateByteArray(array_length):
    byte_array = list([])
    for i in range(0, array_length, 1):
        byte_array.append = random.randint(0,1)
    return byte_array

def encrypt(bytes_array):
    key_array = [1, 1, 1, 1, 1, 1, 1, 1] #old key : [1, 0, 0, 1, 1, 1, 0, 0]
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

def setSampleArrays(sample_size, batch_size=-1):
    xs = None
    ys = None
    if batch_size == -1 or batch_size == None:
        batch_size = sample_size

    if sample_size == 2 and batch_size==sample_size:
        xs = np.array([ encrypt([0, 0]), encrypt([1, 1]) ], dtype=int) #Known encrypted values : 1,0 | 0,1
        ys = np.array([ [0, 0], [1, 1] ], dtype=int)

    if sample_size == 4 and batch_size==sample_size:
        xs = np.array([ encrypt([0,0,0,0]), encrypt([0,1,1,1]), encrypt([0,1,0,1]), encrypt([1,0,1,0]) ], dtype=int)
        ys = np.array([ [0,0,0,0], [0,1,1,1], [0,1,0,1], [1,0,1,0] ], dtype=int)

    if sample_size == 8 and batch_size==sample_size:
        xs = np.array([ encrypt([0, 0, 0, 0, 0, 0, 0, 0]), encrypt([1, 1, 1, 1, 1, 1, 1, 1]), encrypt([1, 0, 0, 0, 1, 1, 1, 1]), encrypt([0, 1, 1, 0, 0, 1, 1, 0]), encrypt([0, 1, 1, 1, 0, 0, 1, 1]) , encrypt([1, 0, 1, 0, 1, 0, 1, 0]), encrypt([0, 1, 0, 1, 0, 1, 0, 1]),  encrypt([1, 1, 1, 1, 0, 0, 0 ,0]) ], dtype=int)
        ys = np.array( [ [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1, 1, 0], [0, 1, 1, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1],  [1, 1, 1, 1, 0, 0, 0 ,0] ], dtype=int)

    if batch_size != sample_size:
        x_data = list([])
        y_data = list([])
        data = None

        for i in range(0, batch_size, 1):
            data = generateByteArray(sample_size)
            x_data.append(encrypt(data))
            y_data.append(data)
        xs = np.array([x_data], dtype=int)
        ys = np.array([y_data], dtype=int)

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
    total_epoch = 0
    number_epochs_training = 50
    networks = list([])

    nb_networks = int(input("Choose how many neural networks you want to create for this simulation : "))

    sample_size = int(input("\n\nPlease enter the size of a batch's sample : "))
    while(sample_size != 2 and sample_size != 4 and sample_size != 8):
        print("/!\ There are currently only three samples sizes available : 2, 4 and 8. /!\ ")
        sample_size = int(input("Please enter the size of a batch's sample : "))

    batch_size_string = input("\n\nPlease enter the number of training samples you want to use.\nEnter \"default\" to use a preset : ")
    batch_size = None
    if batch_size_string == "default":
        batch_size = -1

    xs, ys = setSampleArrays(sample_size, batch_size)
    for i in range(0, nb_networks, 1):
        neural_network_type = int(input("\n\nChoose your type of neural network.\n   1. Classic Neural Network\n   2. Quantum Neural Network\n\nYour choice : "))
        while(neural_network_type != 1 and neural_network_type != 2):
            print("/!\ You did not entered a valid number. Please enter either \"1\" or \"2\". /!\ ")
            neural_network_type = int(input("Choose your type of neural network.\n   1. Classic Neural Network\n   2. Quantum Neural Network\n\nYour choice : "))
        if neural_network_type==1:
            networks.append(NeuralNetwork(xs, ys, sample_size, i+1))
        else:
            networks.append(QuantumNeuralNetwork(xs, ys, sample_size, i+1))

    print("\n\n*-------------------- Training neural networks --------------------*\n\n")
    total_epoch+=number_epochs_training
    for network in networks:
        simulation_beginning = time.time()
        network.trainModel(number_epochs_training)
        time_elapsed = time.time()-simulation_beginning
        print("["+network.name+"] Training time : "+str(float(int(time_elapsed*1000))/1000)+"s")

    while(loop_boolean):
        user_choice = int(input("\n[*] Main menu : \n   [1]. Test the models on a specific entry.\n   [2]. Train the models ("+str(number_epochs_training)+" epochs).\n   [3]. View loss function variation.\n   [4]. Quit the simulation.\n\nYour choice : "))
        while(user_choice != 1 and user_choice != 2 and user_choice != 3 and user_choice != 4):
            print("/!\ You did not entered a valid number. Please enter either \"1\", \"2\", \"3\" or \"4\". /!\ ")
            user_choice = int(input("\n[*] Main menu : \n   [1]. Test the models on a specific entry.\n   [2]. Train the models ("+str(number_epochs_training)+" epochs).\n   [3]. Quit the simulation.\n\nYour choice : "))

        if user_choice == 1:
            user_input = input("\nEnter an array to try (format : 1,0,1,1,1,0,0...) : ")
            value_to_test = user_input.split(',')
            int_array = list([])
            while(len(value_to_test) != sample_size):
                user_input = input("\n/!\ Your input doesn't have a valid length. Please try again. /!\ \nInput : ")
                value_to_test = user_input.split(',')

            for bit in value_to_test:
                int_array.append(int(bit))

            for network in networks:
                network.prediction(int_array)
                garbage_collector = input("\n*--Next--*\n")
            garbage_collector = input("\n----- Press enter to continue. -----\n")

        if user_choice == 2:
            print("\n\n*-------------------- Training neural networks --------------------*\n\n")
            total_epoch+=number_epochs_training
            for network in networks:
                simulation_beginning = time.time()
                network.trainModel(number_epochs_training)
                time_elapsed = time.time()-simulation_beginning
                print("["+network.name+"] Training time : "+str(float(int(time_elapsed*1000))/1000)+"s")
            print()


        if user_choice == 3:
            for network in networks:
                plt.plot(network.training_data_results.history['loss'], label = network.name)
            plt.title("Evolution of the loss function throughout the epochs (current epoch : "+str(total_epoch)+").")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.show(block=False)

        if user_choice == 4:
            print("\n----- Exit -----\n")
            loop_boolean = 0


# -----------------------Quantum Layer initialization------------------- #
"""
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
"""

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
