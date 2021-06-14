import NeuralNetwork as NN
import pennylane as qml
import tensorflow as tf

"""
#------------------------- NETWORK PARAMETERS -------------------------#
"""
input_number_of_layers = 6
input_learning_rate = 0.05
is_strongly_entangled = 0
"""
#----------------------------------------------------------------------#
"""

class QuantumNeuralNetwork(NN.NeuralNetwork):
    def __init__(self, xs_data, ys_data, number_of_qubits=8, number=None, convolutional=0):
        self.name = None
        if number!=None:
            self.name = "Quantum Neural Network n°"+str(number)
        else:
            self.name = "Quantum Neural Network"
        self.xs_data = xs_data
        self.ys_data = ys_data
        self.convolutional = convolutional

        #Quantum Layer creation, featuring basic or strong entanglement between qubits.
        dev = qml.device("default.qubit", wires=number_of_qubits)
        @qml.qnode(dev)
        def qnode(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(number_of_qubits))
            if is_strongly_entangled:
                qml.templates.StronglyEntanglingLayers(weights, wires=range(number_of_qubits))
            else:
                qml.templates.BasicEntanglerLayers(weights, wires=range(number_of_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(number_of_qubits)]
        n_layers = input_number_of_layers
        if is_strongly_entangled:
            weight_shapes = {"weights": (n_layers, number_of_qubits, 3)}
        else:
            weight_shapes = {"weights": (n_layers, number_of_qubits)}
        Quantum_Layer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=number_of_qubits)


        self.model = tf.keras.Sequential([
        Quantum_Layer,
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=input_learning_rate)
        self.model.compile(optimizer=self.optimizer, loss=tf.keras.losses.MeanSquaredError())
