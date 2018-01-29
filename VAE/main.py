"""Semi-supervised learning for k-means and EM for GMM."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from utils import io_tools
from models.kmeans import KMeans


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_iter', 100, 'Number of EM steps to run.')
flags.DEFINE_integer('n_components', 10, 'Number of componenets in kMeans')
flags.DEFINE_string('model_type', 'kmeans', 'gmm or kmeans')


def main(_):
    """High level pipeline.

    This scripts performs the training and evaling and testing stages for
    semi-supervised learning using kMeans algorithm.
    """
    # Load dataset.
    unlabeled_data, _ = io_tools.read_dataset('data/train_no_label.csv')
    n_dims = unlabeled_data.shape[1]

    # Initialize model.
    if FLAGS.model_type == 'kmeans':
        model = KMeans(n_dims, n_components=FLAGS.n_components,
                       max_iter=FLAGS.max_iter)
    else:
        model = GaussianMixtureModel(n_dims, n_components=FLAGS.n_components,
                                     max_iter=FLAGS.max_iter)

    # Unsupervised training.
    model.fit(unlabeled_data)

    # Supervised training.
    train_data, train_label = io_tools.read_dataset(('data/'
                                                     'train_with_label.csv'))
    model.supervised_fit(train_data, train_label)

    # Eval model.
    eval_data, eval_label = io_tools.read_dataset('data/val.csv')
    y_hat_eval = model.supervised_predict(eval_data)
    count=0
    for i in range(len(eval_label)):
        if eval_label[i]==y_hat_eval[i]:
            count+=1
    print(count)
    print(len(y_hat_eval))



    acc = np.sum(y_hat_eval == eval_label) / (1.*eval_data.shape[0])

    print("Accuracy: %s" %acc)

if __name__ == '__main__':
    tf.app.run()
