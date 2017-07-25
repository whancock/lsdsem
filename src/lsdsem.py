import sys
import logging

import numpy as np
import tensorflow as tf

from model import LSDModel
from governor import Governor
from dataset import RocDataset
from embedding import WVEmbedding


class Moo:

    def __init__(self):


        np.random.seed(1234)
        tf.set_random_seed(1234)


        # setup a logger
        logger = logging.getLogger('neural_network')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        handler_stdout = logging.StreamHandler(sys.stdout)
        handler_stdout.setFormatter(formatter)

        logger.addHandler(handler_stdout)
        logger.setLevel(logging.INFO)



        logger.info('Loading data')
        data = RocDataset()
        logger.info('Loading embeddings')
        embedding = WVEmbedding(data)



        train_examples = data.get_good_bad_split(embedding, data.train_data)
        dev_examples = data.get_dev_repr(embedding, data.dev_data)
        test_examples = data.get_dev_repr(embedding, data.test_data)


        model = LSDModel(data, embedding)
        governor = Governor(logger, model)

        governor.train_model(train_examples, dev_examples)
        # governor.test_model(test_examples)




Moo()