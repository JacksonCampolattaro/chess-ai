"""
Implements a Naive-Bayes approach proposed by Antonio Fernandez and Antonio Salmeron.
This approach was developed
"""

import os.path

import numpy as np
from engines.naive_bayes import nb_feature_extractor
from engines.engine import Engine


class FernanNaiveBayesEngine(Engine):

    def __init__(self):
        pass

    def train(self, pgn_file):
        pass

    def choose_move(self, board):
        pass

    def has_model(self, file_name="fs_naive_bayes_model"):
        directory = os.path.dirname(os.path.realpath(__file__))
        return os.path.exists(os.path.join(directory, file_name) + ".npy")

    def save_model(self, file_name="fs_naive_bayes_model"):
        directory = os.path.dirname(os.path.realpath(__file__))
        np.save(os.path.join(directory, file_name), self.model)

    def load_model(self, file_name="fs_naive_bayes_model"):
        directory = os.path.dirname(os.path.realpath(__file__))
        self.model = np.load(os.path.join(directory, file_name) + ".npy", allow_pickle=True)
