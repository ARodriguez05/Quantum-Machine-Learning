import numpy as np
import random
import math

#Create an array of bits of a set length. Integers are used to simulate bits.
def generateByteArray(array_length):
    byte_array = list([])
    for i in range(0, array_length, 1):
        byte_array.append = random.randint(0,1)
    return byte_array

#Convert an integer to a binary array
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

#Binary XOR encryption
def encrypt(bytes_array):
    key_array = [1, 1, 1, 1, 1, 1, 1, 1] #old key : [1, 0, 0, 1, 1, 1, 0, 0]
    encrypted_array = list([])
    for i in range(0,len(bytes_array),1):
        encrypted_array.append((bytes_array[i]+key_array[i])%2)
    return encrypted_array

#Create a dataset based on a sample's size and the length of the batch. If not choosing a custom dataset it'll generate randomly binary arrays.
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
