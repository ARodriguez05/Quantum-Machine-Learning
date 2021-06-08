import os, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.disable(logging.WARNING)

import tensorflow as tf
import cirq
import pennylane as qml

import numpy as np
import matplotlib.pyplot as plt
import random
import time
import math

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
    * Simulation 2 bits [10 epochs] :
        - Réseau classique :
            # learning_rate = 0.025 | [0.08] (??%) | 0.1
            # layers = Dense
        - Réseau quantique :
            # learning_rate = 0.05 (100%)
            # layers = QuantumLayer (6)

    * Simulation 4 bits :
        - Réseau classique :
            # learning_rate = 0.08 (85%) [50 epochs], 0.09 (87.5%) [100 epochs], 0.05 (65%~100%) [100 epochs], 0.03 (82%~90%) [100 epochs]
            # layers = Dense
        - Réseau quantique :
            # learning_rate = 0.05 (75%)
            # layers = QuantumLayer

[*] Faire une démo optimisée de l'apprentissage du XOR 8-bits pour comparer théorie classique et quantique
+ chiffrer temporellement l'efficacité des solutions.

[*] Penser la phase 2 du projet : cryptage sur 16/32 bits, conversion textuelle directe...


[*****] Voir comment implémenter la superposition avec pennylane + Ajouter un convolutional layer pour le classique [*****]
"""


# ---------------------------- Classes --------------------------------- #
class NeuralNetwork:
    def __init__(self, xs_data, ys_data, sample_size=8, number=None, convolutional=0):
        self.name = None
        if number!=None:
            self.name = "Neural Network n°"+str(number)
        else:
            self.name = "Neural Network"
        self.input_shape = (len(xs_data), 1, sample_size)
        self.convolutional = convolutional
        self.total_accuracy = 0.0
        model = list([])

        if convolutional:
            self.name = "Convolutional "+self.name
            self.xs_data = tf.reshape(xs_data, self.input_shape)
            self.ys_data = tf.reshape(ys_data, self.input_shape)
            model.append(tf.keras.layers.Conv1D(filters=8, kernel_size=1, strides=1, input_shape=self.input_shape))
        else :
            self.xs_data = xs_data
            self.ys_data = ys_data

        model.append(tf.keras.layers.Dense(units=sample_size, input_shape=[sample_size]))
        self.model = tf.keras.Sequential(model)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.03)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def trainModel(self, epoch):
        self.training_data_results = self.model.fit(x=self.xs_data, y=self.ys_data, epochs=epoch, verbose=0)

    def prediction(self, bits_table):
        print("\n\n*-----Prediction of "+str(bits_table)+" -----*")
        result = None
        if self.convolutional :
            result = self.model.predict(tf.reshape(bits_table, (1, 1, len(bits_table))))
        else :
            result = self.model.predict([bits_table])
        accurate_value = encrypt(bits_table)
        print("Predicted value : "+str(result[0])+". The correct value is : "+str(accurate_value))

        array = list([])
        for table in result:
            for bit in table:
                if self.convolutional :
                    for extract in bit:
                        value = int(round(extract))
                        if(value<0):
                            value=0
                        array.append(value)
                else :
                    value = int(round(bit))
                    if(value<0):
                        value=0
                    array.append(value)

        accuracy = 0.0
        for i in range(len(bits_table)):
            #print("Valeur array en "+str(i)+" : "+str(array[i])+". Valeur de la table initiale : "+str(bits_table[i])+".\n")
            if array[i]==accurate_value[i]:
                accuracy = accuracy + 1

        accuracy = (accuracy/len(bits_table))*100
        print("Accuracy : "+str(accuracy)+"%")

    def evaluateAccuracyOnSample(self, sample):
        result = None
        if self.convolutional :
            result = self.model.predict(tf.reshape(sample, (1, 1, len(sample))))
        else :
            result = self.model.predict([sample])
        accurate_value = encrypt(sample)

        array = list([])
        for table in result:
            for bit in table:
                if self.convolutional :
                    for extract in bit:
                        value = int(round(extract))
                        if(value<0):
                            value=0
                        array.append(value)
                else :
                    value = int(round(bit))
                    if(value<0):
                        value=0
                    array.append(value)

        accuracy = 0.0
        for i in range(len(sample)):
            if array[i]==accurate_value[i]:
                accuracy = accuracy + 1

        accuracy = (accuracy/len(sample))*100
        return accuracy

    def evaluateTotalAccuracy(self):
        self.total_accuracy = 0
        number_of_samples = len(self.xs_data)*len(self.xs_data)
        for i in range(0, number_of_samples, 1):
            sample = convertToByte(i, len(self.xs_data))
            self.total_accuracy += self.evaluateAccuracyOnSample(sample)
        self.total_accuracy/=number_of_samples
        return self.total_accuracy



class QuantumNeuralNetwork(NeuralNetwork):
    def __init__(self, xs_data, ys_data, number_of_qubits=8, number=None, convolutional=0):
        self.name = None
        if number!=None:
            self.name = "Quantum Neural Network n°"+str(number)
        else:
            self.name = "Quantum Neural Network"
        self.xs_data = xs_data
        self.ys_data = ys_data
        self.convolutional = convolutional

        #Quantum Layer creation
        dev = qml.device("default.qubit", wires=number_of_qubits)
        @qml.qnode(dev)
        def qnode(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(number_of_qubits))
            #qml.templates.StronglyEntanglingLayers(weights, wires=range(number_of_qubits))
            qml.templates.BasicEntanglerLayers(weights, wires=range(number_of_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(number_of_qubits)]
        n_layers = 6 # 1 layer => 71% (15sec) | 10 layers => 50% (88sec)
        weight_shapes = {"weights": (n_layers, number_of_qubits)}
        Quantum_Layer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=number_of_qubits)


        self.model = tf.keras.Sequential([
        Quantum_Layer,
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
        self.model.compile(optimizer=self.optimizer, loss=tf.keras.losses.MeanSquaredError())


# ---------------------------------------------------------------------- #
def generateByteArray(array_length):
    byte_array = list([])
    for i in range(0, array_length, 1):
        byte_array.append = random.randint(0,1)
    return byte_array

def convertToByte(my_integer, byte_length):
    byte_array = list([])
    for i in range(0, byte_length, 1):
        byte_array.append(0)
    pow_2_substract = 64
    while(my_integer>0):
        if(my_integer-pow_2_substract >= 0):
            my_integer-=pow_2_substract
            byte_array[int(math.log(pow_2_substract,2))] = 1
        pow_2_substract/=2
    return byte_array

def encrypt(bytes_array):
    key_array = [1, 1, 1, 1, 1, 1, 1, 1] #old key : [1, 0, 0, 1, 1, 1, 0, 0]
    encrypted_array = list([])
    for i in range(0,len(bytes_array),1):
        encrypted_array.append((bytes_array[i]+key_array[i])%2)
    return encrypted_array

def setSampleArrays(sample_size, batch_size=-1):
    xs = None
    ys = None
    if batch_size == -1 or batch_size == None:
        batch_size = sample_size

    if sample_size == 2 and batch_size==sample_size:
        xs = np.array([ encrypt([0, 0]), encrypt([0, 1]) ], dtype=int) #Known encrypted values : 1,1 | 0,0
        ys = np.array([ [0, 0], [0, 1] ], dtype=int)

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
    number_epochs_training = 100
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
            is_convolutional = int(input("\nShould the network be convolutional ? 1 = yes, 0 = no.\nYour input : "))
            if is_convolutional != 1 and is_convolutional != 0 :
                is_convolutional = 0
            networks.append(NeuralNetwork(xs, ys, sample_size, i+1, is_convolutional))
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
        string_menu = "\n[*] Main menu : \n   [1]. Test the models on a specific entry.\n   [2]. Train the models ("+str(number_epochs_training)+" epochs).\n   [3]. View loss function variation.\n   [4]. Evaluate the accuracy of each network.\n   [5]. Quit the simulation.\n\nYour choice : "
        user_choice = int(input(string_menu))
        while(user_choice != 1 and user_choice != 2 and user_choice != 3 and user_choice != 4 and user_choice != 5):
            print("/!\ You did not entered a valid number. Please enter either \"1\", \"2\", \"3\" or \"4\". /!\ ")
            user_choice = int(input(string_menu))

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
            print()
            for network in networks:
                print("Total accuracy ["+network.name+"] : "+str(network.evaluateTotalAccuracy())+"%")
            garbage_collector = input("\n----- Press enter to continue. -----\n")

        if user_choice == 5:
            print("\n----- Exit -----\n")
            loop_boolean = 0




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
