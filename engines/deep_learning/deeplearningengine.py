import numpy as np
import chess.pgn

from engines.engine import Engine


def extract_features(board):
    # Create a numpy boolean array
    encodings = []
    for piece_type in chess.PIECE_TYPES:
        encodings.append(np.array(board.pieces(piece_type, chess.WHITE).tolist(), np.int8)
                         - np.array(board.pieces(piece_type, chess.BLACK).tolist(), np.int8))

    return np.concatenate(encodings)


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
