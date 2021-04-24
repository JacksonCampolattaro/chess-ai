import random

import numpy as np
import chess.pgn

from engines.engine import Engine
from engines.extractfeatures import interpret_training_data


class DeepLearningEngine(Engine):

    def train(self, pgn_file):
        training_data = interpret_training_data(pgn_file)
        # TODO build & train the model
        pass

    def rank_squares(self, board):
        # TODO Score squares using our model, and then sort them by value
        # return type List(chess.Square)
        return chess.SQUARES

    def rank_moves(self, board: chess.Board, square):
        # TODO Score moves from a square using our model, and then sort them by value
        return board.pseudo_legal_moves

    def choose_move(self, board: chess.Board):

        # Find the list of legal moves
        legal_moves = [move for move in board.legal_moves]

        # Try different squares until we find a legal move
        for square in self.rank_squares(board):

            # Try different moves on this square until we find a legal one
            for move in self.rank_moves(board, square):

                # If we've found a legal move, return that!
                if move in legal_moves:
                    return move

        # If we couldn't find a legal move, return nothing
        return None

    def save_model(self, file_name):
        pass

    def load_model(self, file_name):
        pass
