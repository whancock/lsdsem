import sys
import logging

from model import LSDModel
from dataset import RocDataset
from embedding import WVEmbedding


class Moo:

    def __init__(self):

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




        model = LSDModel(data, embedding)
        model.train_model(train_examples, dev_examples)



Moo()