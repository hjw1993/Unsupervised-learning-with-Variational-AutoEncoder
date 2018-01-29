"""Variation autoencoder."""

import numpy as np
import tensorflow as tf

from tensorflow import contrib
from tensorflow.contrib import layers


class VariationalAutoencoder(object):
    """Varational Autoencoder.
    """
    def __init__(self, ndims=784, nlatent=2):
        """Initializes a VAE

        Args:
            ndims(int): Number of dimensions in the feature.
            nlatent(int): Number of dimensions in the latent space.
        """

        self._ndims = ndims
        self._nlatent = nlatent

        # Create session
        self.session = tf.Session()
        self.x_placeholder = tf.placeholder(tf.float32, [None, ndims])
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [])

        # Build graph.
        self.z_mean, self.z_log_var = self._encoder(self.x_placeholder)
        self.z = self._sample_z(self.z_mean, self.z_log_var)
        self.outputs_tensor = self._decoder(self.z)
        self.ll=self._latent_loss(self.z_mean,self.z_log_var)
        self.rl=self._reconstruction_loss(self.outputs_tensor,self.x_placeholder)

        # Setup loss tensor, predict_tensor, update_op_tensor
        self.loss_tensor = self.loss(self.outputs_tensor, self.x_placeholder,
                                     self.z_mean, self.z_log_var)

        self.update_op_tensor = self.update_op(self.loss_tensor,
                                               self.learning_rate_placeholder)

        # Initialize all variables.
        self.session.run(tf.global_variables_initializer())



    def _sample_z(self, z_mean, z_log_var):
        """Sample z using reparametrize trick.

        Args:
            z_mean (tf.Tensor): The latent mean,
                tensor of dimension (None, 2)
            z_log_var (tf.Tensor): The latent log variance,
                tensor of dimension (None, 2)
        Returns:
            z (tf.Tensor): Random z sampled of dimension (None, 2)
        """


        z = None
        epsilon = tf.random_normal(tf.shape(z_log_var), name='epsilon')

        std_encoder = tf.exp(0.5 * z_log_var)
        z = z_mean+tf.multiply(std_encoder, epsilon)



        return z

    def _encoder(self, x):
        """Encoder block of the network.

        Build a two layer network of fully connected layers, with 100 nodes,
        then 50 nodes. Then two output branches each two 2 nodes representing
        the z_mean and z_log_var.

                             |-> 2 (z_mean)
        Input --> 100 --> 50 -
                             |-> 2 (z_log_var)

        Use activation of tf.nn.softplus.

        Args:
            x (tf.Tensor): The input tensor of dimension (None, 784).
        Returns:
            z_mean(tf.Tensor): The latent mean, tensor of dimension (None, 2).
            z_log_var(tf.Tensor): The latent log variance, tensor of dimension
                (None, 2).

        """

        z_mean = None
        z_log_var = None


        h1 = tf.Variable(self._var_init(784, 100))
        h2 = tf.Variable(self._var_init(100, 50))
        h_m = tf.Variable(self._var_init(50, 2))
        h_log = tf.Variable(self._var_init(50, 2))

        b1 = tf.Variable(tf.zeros([100], dtype=tf.float32))
        b2 = tf.Variable(tf.zeros([50], dtype=tf.float32))
        b_m = tf.Variable(tf.zeros([2], dtype=tf.float32))
        b_log = tf.Variable(tf.zeros([2], dtype=tf.float32))



        layer_1 = tf.nn.softplus(tf.add(tf.matmul(x, h1),b1))
        layer_2 = tf.nn.softplus(tf.add(tf.matmul(layer_1, h2), b2))



        z_mean = tf.add(tf.matmul(layer_2, h_m),b_m)
        z_log_var = tf.add(tf.matmul(layer_2, h_log),b_log)







        return z_mean, z_log_var

    def _decoder(self, z):
        """From a sampled z, decode back into image.

        Build a three layer network of fully connected layers,
        with 50, 100, 784 nodes. Use activation tf.nn.softplus.
        z (2) --> 50 --> 100 --> 784.

        Args:
            z(tf.Tensor): z from _sample_z of dimension (None, 2).
        Returns:
            f(tf.Tensor): Predicted score, tensor of dimension (None, 784).
        """
        d_h1 = tf.Variable(self._var_init(100, 784))
        d_h2 = tf.Variable(self._var_init(50, 100))
        d_h_m = tf.Variable(self._var_init(2, 50))


        d_b1 = tf.Variable(tf.zeros([self._ndims], dtype=tf.float32))
        d_b2 = tf.Variable(tf.zeros([100], dtype=tf.float32))
        d_b_m = tf.Variable(tf.zeros([50], dtype=tf.float32))


        layer_1 = tf.nn.softplus(tf.add(tf.matmul(self.z, d_h_m),d_b_m))
        layer_2 = tf.nn.softplus(tf.add(tf.matmul(layer_1, d_h2),d_b2))

        f = None
        f = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, d_h1),d_b1))


        return f

    def _latent_loss(self, z_mean, z_log_var):
        """Constructs the latent loss.

        Args:
            z_mean(tf.Tensor): Tensor of dimension (None, 2)
            z_log_var(tf.Tensor): Tensor of dimension (None, 2)

        Returns:
            latent_loss: Tensor of dimension (None,). Sum the loss over
            dimension 1.
        """

        latent_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), 1)


        return latent_loss

    def _reconstruction_loss(self, f, y):
        """Constructs the reconstruction loss, assume Gaussian.
        Args:
            f(tf.Tensor): Predicted score for each example, dimension (None,
                784).
            y(tf.Tensor): Ground truth for each example, dimension (None, 784)
        Returns:
            Tensor for dimension (None,). Sum the loss over dimension 1.
        """
        reconstr_loss =-tf.reduce_sum(y * tf.log(1e-8+f)+ (1 - y) * tf.log(1e-8+1 - f),1)

        return reconstr_loss

    def loss(self, f, y, z_mean, z_var):
        """Computes the total loss.

        Computes the sum of latent and reconstruction loss then average over
        dimension 0.

        Returns:
            (1) averged loss of latent_loss and reconstruction loss over
                dimension 0.
        """
        latent_loss=self._latent_loss(z_mean,z_var)
        r_loss=self._reconstruction_loss(f,y)
        loss_tensor = tf.reduce_mean(r_loss + latent_loss)


        return loss_tensor

    def update_op(self, loss, learning_rate):
        """Creates the update optimizer.

        Use tf.train.AdamOptimizer to obtain the update op.

        Args:
            loss: Tensor containing the loss function.
            learning_rate: A scalar, learning rate for gradient descent.
        Returns:
            (1) Update opt tensorflow operation.
        """
        update_op_tensor = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder).minimize(self.loss_tensor)
        return update_op_tensor

    def generate_samples(self, z_np):
        """Generates a random sample from the provided z_np.

        Args:
            z_np(numpy.ndarray): Numpy array of dimension (batch_size, 2).

        Returns:
            (1) The sampled images (numpy.ndarray) of dimension (batch_size,
                748).
        """





        return self.session.run(self.outputs_tensor,feed_dict={self.z: z_np})

    def _var_init(self ,fan_in, fan_out, constant=1):

        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out),
                                 minval=low, maxval=high,
                                 dtype=tf.float32)



