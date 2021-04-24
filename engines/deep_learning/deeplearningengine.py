import os

import chess.pgn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engines.engine import Engine
from engines.extractfeatures import interpret_training_data, extract_features


class DeepLearningEngine(Engine):

    def __init__(self):
        self.learning_rate = 0.0015

        # The piece chooser network selects the location of the piece to pick up from the board
        self.piece_chooser = torch.nn.Sequential(
            torch.nn.Linear(384, 786),
            torch.nn.ReLU(),
            torch.nn.Linear(786, 64),
        )

        # The piece placer networks select the location to put down the piece
        # (A different network is chosen depending on the piece's type)
        self.piece_placers = {
            # Each piece type is mapped to its own neural network
            piece:
                torch.nn.Sequential(
                    torch.nn.Linear(384, 786),
                    torch.nn.ReLU(),
                    torch.nn.Linear(786, 64),
                )
            for piece in chess.PIECE_TYPES
        }

    def train(self, pgn_file):
        random_data = torch.rand(384)
        print(self.piece_chooser(random_data))
        # training_data = interpret_training_data(pgn_file, 5)
        #
        # for board_state, move_choices in training_data.items():
        #     board_state_decoded = extract_features(chess.Board(board_state))
        #     piece_chosen = move_choices[0][0]
        #
        #     print(board_state_decoded)
        #     print(piece_chosen)
        #
        # # TODO build & train the model
        # pass

    def choose_move(self, board: chess.Board):
        return None

    def save_model(self, file_name="deep_learning_model"):
        directory = os.path.dirname(os.path.realpath(__file__))
        torch.save(self.piece_chooser, os.path.join(directory, file_name))

    def load_model(self, file_name="deep_learning_model"):
        directory = os.path.dirname(os.path.realpath(__file__))
        self.piece_chooser = torch.load(os.path.join(directory, file_name))
