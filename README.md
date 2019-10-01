The project is a basic neural network implementation with 3 hidden layers for MNSIT digit recognition using Keras and tensorflow.
Mnist dataset contains images with 28*28 pixels.
Input layer size is 784 = 28*28 and output layer size is 10 which correspons to 10 classes from 0-9, where the result of the neural network is a vector contatining the probbaility of input image being a particular class 

eg:[0, 0.01, 0.02, 0.39, 0, 0, 0.03, 0, 0.25,0.3]. The result would be 3.

The project achieves an accuracy of 96.32% with epochs = 10 and an ccuracy of approx 95% with epochs= 20.
