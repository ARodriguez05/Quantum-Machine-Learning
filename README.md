# Quantum-Machine-Learning

  This project features methods to use simple classic and quantum neural networks to try their efficiency when it comes to decipher a database. Using various libraries such as Tensorflow and Pennylane, it showcases how far a neural network can learn a decryption method based on ciphered and clear data assumed leaked from a database.

## Features

___How to use the simulator :___ To run the simulator, just launch the file "main.py" in any environment supporting Python3. You might need to install many libraries in your environment such as Tensorflow or Pennylane, please refer to their respective tutorials to prepare your setup. You'll be asked several parameters before running the simulation :
  * __Number of neural networks :__ You can either create one or several neural networks if you want to compare them on a specific dataset. You can modify their parameters (learning rate, loss function...) in their respective files "NeuralNetwork.py" and "QuantumNeuralNetwork.py" in the constructor of each class.

  * __Size of a batch's sample :__ The size/length of one sample of your training dataset. Datasets must have the same length. If you want to try the decryption on a dataset of 4-bits arrays, you'll enter 4.

  * __Number of training samples :__ The number of samples you want to use for training. For binary encryption/decryption, you can use a built-in function to generate your binary arrays at random, specific arrays or use built-in datasets and see whereas your dataset contains too much data or not enough for your networks.

  * __Type of neural network :__ This is where your choose which neural network you want to simulate. It can either be a classic or quantum neural network and, in the case of a classic neural network, a regular one with a Dense layer only or a Convolutional network.

## Results


You'll find published here the current results for n-bits binary decryption with the parameters required to achieve such results.

---

  * __2-bits XOR decryption :__ Encryption was done over 100 epochs with the following key -> [1,0]. The dataset was the 2-bit built-in preset.
     * _Classic Networks specs_
         * loss function : Mean Squared
         * optimizer : Adam (learning rate = 0.08)
         * accuracy : 75%

     * _Convolutional Networks specs_
         * loss function : Mean Squared
         * optimizer : Adam (learning rate = 0.08)
         * number of filters : 8
         * accuracy : 75%

     * _Quantum Networks specs_
         * loss function : Mean Squared
         * optimizer : Adam (learning rate = 0.05)
         * accuracy : 100%

<img src="https://user-images.githubusercontent.com/72410925/121705960-7681e080-cad5-11eb-8ea8-2473997f1b99.png">

---
  * __4-bits XOR decryption :__ Encryption was done over 100 epochs with the following key -> [1, 0, 1, 0]. The dataset was the 4-bit built-in preset.

     * _Classic Networks specs_
         * loss function : Mean Squared
         * optimizer : Adam (learning rate = 0.07)
         * training time : 0.54s
         * accuracy : 80~100%

     * _Convolutional Networks specs_
         * loss function : Mean Squared
         * optimizer : Adam (learning rate = 0.07)
         * training time : 0.567s
         * number of filters : 10
         * accuracy : 88%

     * _Quantum Networks specs_
         * loss function : Mean Squared
         * optimizer : Adam (learning rate = 0.04)
         * training time : 64.315s
         * number of layers : 6
         * accuracy : 72%
<img src="https://user-images.githubusercontent.com/72410925/121704259-de372c00-cad3-11eb-8bde-00c57192057f.png">

---
  * __8-bits XOR decryption :__ Encryption was done over 100 to 200 epochs with the following key -> [1, 0, 0, 1, 1, 1, 0, 0]. The dataset was the 8-bit built-in preset.
     * _Classic Networks specs_
         * loss function : Mean Squared
         * optimizer : Adam (learning rate = 0.07)
         * training time : ~0.6s
         * accuracy : 82~88%

     * _Convolutional Networks specs_
         * loss function : Mean Squared
         * optimizer : Adam (learning rate = 0.07)
         * training time : ~0.5s
         * number of filters : 10
         * accuracy : 85~91%

     * _Quantum Networks specs_
         * loss function : Mean Squared
         * optimizer : Adam (learning rate = 0.06)
         * training time : ~300s
         * number of layers : 7
         * accuracy : 78%
<img src="https://user-images.githubusercontent.com/72410925/121707141-9239b680-cad6-11eb-8302-6ff5ab7cfc77.png">
