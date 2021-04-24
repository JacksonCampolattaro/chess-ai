import numpy as np
import chess.pgn

from engines.engine import Engine
from engines.extractfeatures import interpret_training_data


class DeepLearningEngine(Engine):

    def train(self, pgn_file):
        training_data = interpret_training_data(pgn_file)
        # TODO build & train the model
        pass

    def choose_square(self, board):
        # TODO Find the appropriate square, based on the current board state
        # return type chess.Square()
        pass

    def choose_move(self, board):
        pass

    def save_model(self, file_name):
        pass

    def load_model(self, file_name):
        pass
