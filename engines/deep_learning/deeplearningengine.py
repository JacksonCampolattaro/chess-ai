import os

import chess.pgn

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from engines.engine import Engine
from engines.extractfeatures import interpret_training_data, extract_features


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # First fully connected layer (which takes in our decoded board state)
        self.fc1 = nn.Linear(384, 384)

        # Second fully connected layer (which outputs the selected board location)
        self.fc2 = nn.Linear(384, 64)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    pass


class DeepLearningEngine(Engine):

    def __init__(self):
        self.learning_rate = 0.0015

        # The piece chooser network selects the location of the piece to pick up from the board
        self.piece_chooser = Net()

        # The piece placer networks select the location to put down the piece
        # (A different network is chosen depending on the piece's type)
        self.piece_placers = {piece: Net() for piece in chess.PIECE_TYPES}

    def train(self, pgn_file):
        training_data = interpret_training_data(pgn_file, 5)

        for board_state, move_choices in training_data.items():
            board_state_decoded = extract_features(chess.Board(board_state))
            piece_chosen = move_choices[0][0]

            print(board_state_decoded)
            print(piece_chosen)

        # TODO build & train the model
        pass

    def choose_move(self, board: chess.Board):
        return None

    def save_model(self, file_name="deep_learning_model"):
        directory = os.path.dirname(os.path.realpath(__file__))
        np.save(os.path.join(directory, file_name), self.model)

    def load_model(self, file_name="deep_learning_model"):
        directory = os.path.dirname(os.path.realpath(__file__))
        self.model = np.load(os.path.join(directory, file_name) + ".npy", allow_pickle=True)
