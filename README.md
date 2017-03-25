# mnist

# Normal MNIST Linear
Simple  works with ~90% accuracy using normal gradient descent optimizer and a learning rate of 0.5.

# Negative Images MNIST Linear
Negative images work with ~90% accuracy using adam gradient descent.
To train negative images with normal gradient descent optimizer you need a very low learning rate and a lot of time to train.

# Combined MNIST ( Normal + Negative Images) Linear
A simple Linear classifier doesn't converge for both negative and normal images.
This is not a linearly separateable scenario
This makes sense, because there is only 1 weight per input node matching to an output node.
That 1 weight can't have two opposing influences over the output node.


Tensorflow Version 1.0.1