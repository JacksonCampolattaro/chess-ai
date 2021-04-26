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


def onehot_board(square: chess.Square):
    board_encoding = np.zeros(shape=64, dtype=np.int8)
    board_encoding[square] = 1.0
    return board_encoding
    # return np.reshape(board_encoding, (8, 8))


def as_bitboard(board: chess.Board, piece: chess.PieceType, color: chess.Color):
    return np.reshape(np.asarray(board.pieces(piece, color).tolist(), np.int8), (8, 8))


def extract_features_3d(board: chess.Board):
    return np.asarray([
        (as_bitboard(board, piece, chess.WHITE) - as_bitboard(board, piece, chess.BLACK))
        for piece in chess.PIECE_TYPES
    ])


def move_as_entry(board, move):
    # Get the feature set from the board
    features = extract_features_3d(board)

    # Determine the labels
    labels = [move.from_square, None, None, None, None, None, None]
    labels[board.piece_at(move.from_square).piece_type] = move.to_square

    # Combine the results
    return features, labels


def interpret_data(pgn_file, count: int, color: chess.Color):
    pgn = open(pgn_file)

    dataset = []

    # Iterate over games in the training data
    game = chess.pgn.read_game(pgn)
    while game is not None:

        board = game.board()

        # Determine the outcome of the game
        game_result = game.headers["Result"]
        winner = chess.WHITE if (game_result == "1-0") else chess.BLACK if (game_result == "0-1") else None

        # Only train on winning games
        if winner == color:

            # Iterate over moves in the training data
            turn = chess.WHITE
            for move in game.mainline_moves():

                # Only train on moves by the relevant color
                if turn == color:
                    dataset.append(move_as_entry(board, move))

                    # If we've generated enough data, return the dataset
                    if len(dataset) >= count:
                        return dataset

                # Apply this move to the board
                board.push(move)

                # Alternating turns
                turn = not turn

        # Get the next game
        game = chess.pgn.read_game(pgn)

    return dataset


# From
# https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks
def chunks(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


class DeepLearningEngine(Engine):

    def __init__(self):
        self.input_encoding_size = 384
        self.output_encoding_size = 64

        self.batch_size = 100

        self.learning_rate = 0.0015
        self.weight_decay = 0.999
        # self.loss_function = torch.nn.CrossEntropyLoss()
        self.loss_function = torch.nn.NLLLoss()

        hidden_layer_size = 128

        # The piece chooser network selects the location of the piece to pick up from the board
        self.piece_chooser = torch.nn.Sequential(
            torch.nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.Flatten(),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, hidden_layer_size),
            torch.nn.Linear(hidden_layer_size, 64),
            torch.nn.LogSoftmax(dim=0),
        )

        # The piece placer networks select the location to put down the piece
        # (A different network is chosen depending on the piece's type)
        self.piece_placers = {
            # Each piece type is mapped to its own neural network
            piece:
                torch.nn.Sequential(
                    torch.nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1),
                    torch.nn.Flatten(),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2048, hidden_layer_size),
                    torch.nn.Linear(hidden_layer_size, 64),
                    torch.nn.LogSoftmax(dim=0),
                )
            for piece in [0] + list(chess.PIECE_TYPES)
        }

    def train_nets(self, training_data, piece):

        # Train the relevant net
        net = self.piece_placers[piece]

        # This optimizer will do gradient descent for us
        optimizer = optim.RMSprop(net.parameters(), lr=self.learning_rate,
                                  weight_decay=self.weight_decay)

        # Training is done in batches
        running_loss = 0.0
        for i, batch in enumerate(chunks(training_data, self.batch_size)):

            # Extract features and relevant labels from the training dataset
            features = [features for features, labels in batch]
            labels = [labels[piece] for features, labels in batch]

            # Convert features and labels to tensors
            x = torch.Tensor(features)
            y = torch.tensor(labels, dtype=torch.long)

            # Reset gradients for gradient descent
            optimizer.zero_grad()

            # Attempt to use the piece chooser model to select a location (forward propegation)
            pred_y: torch.Tensor = net(x)

            # Apply the loss function, comparing predicted values to actual
            loss = self.loss_function(pred_y, y)

            # Backpropagate, and then update weights
            loss.backward()
            optimizer.step()

            # print(x, y, pred_y)
            # print(pred_y.detach().numpy())

            running_loss += loss.item() / len(batch)
            if i % 100 == 99:
                print(running_loss)
                running_loss = 0

    def train(self, pgn_file):
        # training_data = interpret_training_data(pgn_file, 5000)
        # self.train_piece_chooser(training_data)

        dataset = interpret_data(pgn_file, 1000000, chess.WHITE)
        print(len(dataset))

        # Train the piece chooser
        print("Training piece chooser")
        self.train_nets(dataset, 0)

        # Train each piece placer
        for piece in chess.PIECE_TYPES:
            print(f"Training piece placer for {chess.piece_name(piece)}")
            relevant_dataset = [(features, labels) for features, labels in dataset if labels[piece] is not None]
            self.train_nets(relevant_dataset, piece)

        print("done")

        # TODO build & train the model

    def choose_move(self, board: chess.Board):
        return None

    def save_model(self, file_name="deep_learning_model"):
        directory = os.path.dirname(os.path.realpath(__file__))
        torch.save(self.piece_chooser, os.path.join(directory, file_name))

    def load_model(self, file_name="deep_learning_model"):
        directory = os.path.dirname(os.path.realpath(__file__))
        self.piece_chooser = torch.load(os.path.join(directory, file_name))
