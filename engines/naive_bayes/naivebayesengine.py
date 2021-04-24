"""
This approach was developed in response to the old Naive-Bayes implementation
"""

import os.path

import numpy as np
import chess
from engines import extractfeatures
from engines.engine import Engine

train_limit = 300


class NaiveBayesEngine(Engine):

    def __init__(self):
        self.move_select_model = []
        self.pawn_model = []
        self.knight_model = []
        self.bishop_model = []
        self.rook_model = []
        self.queen_model = []
        self.king_model = []

    def train(self, pgn_file):
        # Extract data from pgn file using extractfeatures.py
        move_dict = extractfeatures.interpret_training_data(pgn_file, train_limit)
        # Set up counting vectors
        labelCount = np.zeros([7, 64], dtype=int)
        conditionalFeatureCount = np.zeros([7, 64, 768], dtype=int)
        # Go through each entry in the dictionary and add respective values into count vectors
        for board_key in move_dict:
            board = chess.Board(board_key)
            feature_set = extractfeatures.extract_features_sparse(board)
            fs_indexes = np.argwhere(feature_set == 1)
            for i in range(7):
                for label in move_dict[board_key][i]:
                    labelCount[i, label] += 1
                    conditionalFeatureCount[i, label, fs_indexes] += 1

        prior_prob = [row / np.sum(row) for row in labelCount]
        conditional_prob = [np.transpose(column / np.sum(column)) for column in
                            [np.transpose(table) for table in conditionalFeatureCount]]
        # conditional_prob = conditionalFeatureCount / np.sum(conditionalFeatureCount, axis=2)[:, :, np.newaxis]

        # Store data into the models
        self.move_select_model = self.prep_model(prior_prob[0], conditional_prob[0])
        self.pawn_model = self.prep_model(prior_prob[1], conditional_prob[1])
        self.knight_model = self.prep_model(prior_prob[2], conditional_prob[2])
        self.bishop_model = self.prep_model(prior_prob[3], conditional_prob[3])
        self.rook_model = self.prep_model(prior_prob[4], conditional_prob[4])
        self.queen_model = self.prep_model(prior_prob[5], conditional_prob[5])
        self.king_model = self.prep_model(prior_prob[6], conditional_prob[6])

    def prep_model(self, prior_prob, conditional_prob):
        model = np.zeros([64, 768 + 1])
        model[:, 0] = prior_prob
        model[:, 1:] = conditional_prob
        return model

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


if __name__ == '__main__':
    nb = NaiveBayesEngine()
    nb.train("lichess_db_standard_rated_2013-01.pgn")
    print(nb.move_select_model)
