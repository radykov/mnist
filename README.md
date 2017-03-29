# Normal MNIST Linear
Simple  works with ~90% accuracy using normal gradient descent optimizer and a learning rate of 0.5.

# Negative Images MNIST Linear
Negative images work with ~90% accuracy using adam gradient descent.
To train negative images with normal gradient descent optimizer you need a very low learning rate and a lot of time to train.

# Combined MNIST ( Normal + Negative Images) Linear
A simple Linear classifier gets normal images right only 55% of the time and negative images about 45%
The trick is that negative images appear to be harder to train, so more of the data being trained is negative.
As opposed to having 50% of the training on negative images

# Combined MNIST (Normal + Negative Images) MLP
A simple MLP with 1 input, 1 hidden and 1 output.
20 hidden nodes,
64% of the time correct for normal
67% correct for negative images

# CNN MNIST
Gets to 95% accuracy within 1,000 iterations and batch of 100

# CNN MNIST Negative
Gets to 94% accuracy within 1,000 iterations and batch of 100

# CNN MNIST Normal + Negative
Gets 92% accuracy within 1,000 iterations and a batch of 100 for both normal and negative images
This is much closer performance to the seperated data sets than the other approaches had.




# Tensorflow & Python versions
Tensorflow Version 1.0.1
Python version 2.7