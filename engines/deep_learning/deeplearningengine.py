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
        self.input_encoding_size = 384
        self.output_encoding_size = 64

        self.num_epochs = 2
        self.batch_size = 100

        self.learning_rate = 0.0015
        self.loss_function = torch.nn.NLLLoss()

        hidden_layer_size = self.input_encoding_size * 2

        # The piece chooser network selects the location of the piece to pick up from the board
        self.piece_chooser = torch.nn.Sequential(
            torch.nn.Linear(self.input_encoding_size, hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_size, self.output_encoding_size),
            torch.nn.Softmax(),
        )

        # The piece placer networks select the location to put down the piece
        # (A different network is chosen depending on the piece's type)
        self.piece_placers = {
            # Each piece type is mapped to its own neural network
            piece:
                torch.nn.Sequential(
                    torch.nn.Linear(self.input_encoding_size, hidden_layer_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_layer_size, self.output_encoding_size),
                )
            for piece in chess.PIECE_TYPES
        }

    def train_piece_chooser(self, training_data):
        for board_state, move_choices in training_data.items():
            board_state_decoded = extract_features(chess.Board(board_state))
            pieces_chosen = move_choices[0]

            x = torch.from_numpy(board_state_decoded)
            y = pieces_chosen

            print(x)
            print(y)

        # While there's still data
        # Get a batch of inputs and labels

        # https: // pytorch.org / tutorials / beginner / blitz / cifar10_tutorial.html

        # Forward propegation
        # Loss
        # backpropegation

        # Occasionally print data

        pass

    def train_piece_placer(self, training_data, piece: chess.PieceType):
        pass

    def train(self, pgn_file):
        training_data = interpret_training_data(pgn_file, 5)
        self.train_piece_chooser(training_data)

        # TODO build & train the model

    def choose_move(self, board: chess.Board):
        return None

    def save_model(self, file_name="deep_learning_model"):
        directory = os.path.dirname(os.path.realpath(__file__))
        torch.save(self.piece_chooser, os.path.join(directory, file_name))

    def load_model(self, file_name="deep_learning_model"):
        directory = os.path.dirname(os.path.realpath(__file__))
        self.piece_chooser = torch.load(os.path.join(directory, file_name))
