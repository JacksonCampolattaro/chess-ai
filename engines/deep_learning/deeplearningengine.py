import numpy as np
import chess.pgn

from engines.engine import Engine
from engines.extractfeatures import extract_features


class DeepLearningEngine(Engine):

    def train(self, pgn_file):
        pgn = open(pgn_file)

        for i in range(100):
            game = chess.pgn.read_game(pgn)
            board = game.board()
            print(extract_features(board).size)
        pass

    def choose_move(self, board):
        pass

    def save_model(self, file_name):
        pass

    def load_model(self, file_name):
        pass
