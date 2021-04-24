import os.path

import numpy as np
from engines.naive_bayes import nb_feature_extractor
from engines.engine import Engine


class NaiveBayesEngine(Engine):

    def __init__(self):
        self.model = np.array([])
        self.num_trained_moves = 0
        self.num_total_moves = 0

    def train(self, pgn_file):
        # Uses feature extractor to get train data from pgn file, then trains from dictionary
        move_dict = nb_feature_extractor.nb_extract_moves(pgn_file)
        self.train_from_dict(move_dict)

    def train_from_dict(self, move_dictionary):
        # Convert count dictionary into Naive-Bayes probability matrix
        # NB Model Format
        #   0           0           "Loss"            "Win"
        #   0           0           P(Y=Loss)         P(Y=Win)
        #   move0       P(X=move0)  P(X=move0|Y=Loss) P(X=move0|Y=Win)
        #   move1       P(X=move1)  P(x=move1|Y=Loss) P(X=move1|Y=Win)
        #   ...
        #   moveN       P(X=moveN)  P(x=moveN|Y=Loss) P(X=moveN|Y=Win)

        data = np.array(list(move_dictionary.items()), dtype=object)
        data = np.array([[entry[0], entry[1][0], entry[1][1]] for entry in data]).flatten().reshape(-1, 3)
        winning_move_count = np.sum(np.array(data[:, 1:], dtype=int), axis=0)
        self.num_total_moves = np.sum(winning_move_count)
        PXY = np.divide(np.array(data[:, 1:], dtype=int), winning_move_count)
        PX = np.divide(np.sum(np.array(data[:, 1:], dtype=int), axis=1), self.num_total_moves)
        PY = np.divide(winning_move_count, self.num_total_moves)
        # Format model's np matrix
        self.model = np.zeros([2 + data.shape[0], 4], dtype=object)
        self.model[0, [2, 3]] = ["Loss", "Win"]
        self.model[2:, 0] = data[:, 0]
        self.model[1, [2, 3]] = PY
        self.model[2:, [2, 3]] = PXY
        self.model[2:, 1] = PX

    def choose_move(self, board):
        #  Given a board state, generate all legal moves using python-chess, then calculate the move that gives
        #  the highest win probability
        legal_moves = board.legal_moves
        scores = []
        for move in legal_moves:
            key = move.uci()
            if key not in self.model[2:, 0]:
                self.num_trained_moves += 1
                self.num_total_moves += 1
                PX = 1 / self.num_total_moves
                PXY = [1 / self.num_trained_moves, 1 / self.num_trained_moves]
                np.append(self.model, [key, PX, PXY])
                # print(self.model)
            index = np.argwhere(self.model[:, 0] == key)
            PYX = (self.model[1, 3] * self.model[index, 3]) / (self.model[index, 1])
            scores.append(PYX)
        return list(legal_moves)[np.argmax(scores)]

    def has_model(self, file_name="naive_bayes_model"):
        directory = os.path.dirname(os.path.realpath(__file__))
        return os.path.exists(os.path.join(directory, file_name) + ".npy")

    def save_model(self, file_name="naive_bayes_model"):
        directory = os.path.dirname(os.path.realpath(__file__))
        np.save(os.path.join(directory, file_name), self.model)

    def load_model(self, file_name="naive_bayes_model"):
        directory = os.path.dirname(os.path.realpath(__file__))
        self.model = np.load(os.path.join(directory, file_name) + ".npy", allow_pickle=True)
