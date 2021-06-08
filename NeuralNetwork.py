import tensorflow as tf
import functions as toolkit

class NeuralNetwork:

    #Class constructor
    def __init__(self, xs_data, ys_data, sample_size=8, number=None, convolutional=0):
        self.name = None
        if number!=None:
            self.name = "Neural Network n°"+str(number)
        else:
            self.name = "Neural Network"
        self.input_shape = (len(xs_data), 1, sample_size)
        self.convolutional = convolutional #Parameter used to handle the data based on the input_shape, which will be different for a convolutional network.
        self.total_accuracy = 0.0
        model = list([])

        #Reshaping the input data and adding a convolutional layer if required.
        if convolutional:
            self.name = "Convolutional "+self.name
            self.xs_data = tf.reshape(xs_data, self.input_shape)
            self.ys_data = tf.reshape(ys_data, self.input_shape)
            model.append(tf.keras.layers.Conv1D(filters=8, kernel_size=1, strides=1, input_shape=self.input_shape))
        else :
            self.xs_data = xs_data
            self.ys_data = ys_data

        #Regular Dense layer for data processing.
        model.append(tf.keras.layers.Dense(units=sample_size, input_shape=[sample_size]))
        self.model = tf.keras.Sequential(model)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.03)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    #Function used to train the model over a number of epochs and save the training history.
    def trainModel(self, epoch):
        self.training_data_results = self.model.fit(x=self.xs_data, y=self.ys_data, epochs=epoch, verbose=0)

    #Prediction of an input value, here a bit array such as [1, 0, 1, 0] to represent the binary value 1010.
    #---
    #--- The encryption method being symmetrical, we use the encryption method
    #--- to get the correct value the network should have after decryption.
    #--- We can evaluate the network's accuracy this way by comparing to the network's output the correct value.
    def prediction(self, bits_table, get_output=0, verbose=1):
        if verbose :
            print("\n\n*-----Prediction of "+str(bits_table)+" -----*")
        result = None
        if self.convolutional :
            result = self.model.predict(tf.reshape(bits_table, (1, 1, len(bits_table))))
        else :
            result = self.model.predict([bits_table])
        accurate_value = toolkit.encrypt(bits_table)
        if verbose :
            print("Predicted value : "+str(result[0])+". The correct value is : "+str(accurate_value))

        #Processing the predicted values to reshape them into a binary array.
        array = list([])
        for table in result: #Data format can either be [[0.98546, 0.003135]] or [[[0.98546, 0.003135]]]. Output => [1, 0] in that example.
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
            if array[i]==accurate_value[i]:
                accuracy = accuracy + 1

        accuracy = (accuracy/len(bits_table))*100
        if verbose :
            print("Accuracy : "+str(accuracy)+"%")
        if get_output :
            return accuracy

    #Get the overall accuracy of the network on the database. For binary arrays it generates all the possible arrays.
    def evaluateTotalAccuracy(self):
        self.total_accuracy = 0
        number_of_samples = len(self.xs_data)*len(self.xs_data)
        for i in range(0, number_of_samples, 1):
            sample = toolkit.convertToByte(i, len(self.xs_data))
            self.total_accuracy += self.prediction(sample, get_output=1,verbose=0)
        self.total_accuracy/=number_of_samples
        return self.total_accuracy
