import os
from itertools import zip_longest

import chess.pgn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from engines.engine import Engine
from engines.extractfeatures import interpret_training_data, extract_features, create_bitboard, create_floatboard


# From
# https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks
def chunks(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

class DeepLearningEngine(Engine):

    def __init__(self):
        self.input_encoding_size = 384
        self.output_encoding_size = 64

        self.batch_size = 10

        self.learning_rate = 0.0015
        # self.loss_function = torch.nn.CrossEntropyLoss()
        self.loss_function = torch.nn.L1Loss()

        hidden_layer_size = self.input_encoding_size * 2

        # The piece chooser network selects the location of the piece to pick up from the board
        self.piece_chooser = torch.nn.Sequential(
            torch.nn.Linear(self.input_encoding_size, hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_size, self.output_encoding_size),
            # torch.nn.Softmax(),
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

        # This optimizer will do gradient descent for us
        optimizer = optim.SGD(self.piece_chooser.parameters(), lr=self.learning_rate, momentum=0.9)

        # Training is done in batches
        for batch in chunks(list(training_data.items()), self.batch_size):

            # Extract features and relevant labels from the training dataset
            features = [extract_features(chess.Board(board)) for board, results in batch]
            labels = [create_floatboard(results[0]) for board, results in batch]

            # Convert features and labels to tensors
            x = torch.Tensor(features)
            y = torch.Tensor(labels)

            # Reset gradients for gradient descent
            optimizer.zero_grad()

            # Attempt to use the piece chooser model to select a location (forward propegation)
            pred_y = self.piece_chooser(x)

            # Apply the loss function, comparing predicted values to actual
            loss = self.loss_function(pred_y, y)

            # Backpropagate, and then update weights
            loss.backward()
            optimizer.step()

            print(loss.item())

            # print(x, y, pred_y)

    def train_piece_placer(self, training_data, piece: chess.PieceType):
        pass

    def train(self, pgn_file):
        training_data = interpret_training_data(pgn_file, 5000)
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
