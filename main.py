import os, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.disable(logging.WARNING)

import matplotlib.pyplot as plt
import time

import numpy as np
import QuantumNeuralNetwork as QNN
import NeuralNetwork as NN
import functions as toolkit

np.random.seed(1234)

#WIP
def Overfitting_Example():
    sample_size = 8
    batch_size = 40
    xs, ys = toolkit.setSampleArrays(sample_size, batch_size)

    xs_training = xs[0:int(round(len(xs)*0.2))]
    xs_verif = xs[int(round(len(xs)*0.2)):len(xs)]

    ys_training = ys[0:int(round(len(ys)*0.2))]
    ys_verif = ys[int(round(len(ys)*0.2)):len(ys)]

    network = NN.NeuralNetwork(xs_training, ys_training, sample_size, convolutional=0)
    network.trainModel(100)

    plt.plot(network.training_data_results.history['loss'], label = network.name+" [Training]")
    plt.title("Evolution of the loss function throughout the epochs.")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    accuracy = 0.0
    for sample in xs_training:
        print(sample)
        accuracy+=network.prediction(sample, get_output=1, verbose=0)
    accuracy/=(len(xs_training)*0.01)
    print("Accuracy on training sample : "+str(accuracy)+"%")

    accuracy = 0.0
    for sample in xs_verif:
        accuracy+=network.prediction(sample, get_output=1, verbose=0)
    accuracy/=(len(xs_verif)*0.01)
    print("Accuracy on training sample : "+str(accuracy)+"%")
    network.evaluateTotalAccuracy()

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
    """
    while(sample_size != 2 and sample_size != 4 and sample_size != 8):
        print("/!\ There are currently only three samples sizes available : 2, 4 and 8. /!\ ")
        sample_size = int(input("Please enter the size of a batch's sample : "))
    """
    batch_size_string = input("\n\nPlease enter the number of training samples you want to use.\nEnter \"default\" to use a preset : ")
    batch_size = None
    if batch_size_string == "default":
        batch_size = -1
    else :
        batch_size = int(batch_size_string)

    xs, ys = toolkit.setSampleArrays(sample_size, batch_size)
    for i in range(0, nb_networks, 1):
        neural_network_type = int(input("\n\nChoose your type of neural network.\n   1. Classic Neural Network\n   2. Quantum Neural Network\n\nYour choice : "))
        while(neural_network_type != 1 and neural_network_type != 2):
            print("/!\ You did not entered a valid number. Please enter either \"1\" or \"2\". /!\ ")
            neural_network_type = int(input("Choose your type of neural network.\n   1. Classic Neural Network\n   2. Quantum Neural Network\n\nYour choice : "))
        if neural_network_type==1:
            is_convolutional = int(input("\nShould the network be convolutional ? 1 = yes, 0 = no.\nYour input : "))
            if is_convolutional != 1 and is_convolutional != 0 :
                is_convolutional = 0
            networks.append(NN.NeuralNetwork(xs, ys, sample_size, i+1, is_convolutional))
        else:
            networks.append(QNN.QuantumNeuralNetwork(xs, ys, sample_size, i+1))

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
