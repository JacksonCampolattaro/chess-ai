import chess
import numpy as np


def extract_features(board):
    encodings = []
    for piece_type in chess.PIECE_TYPES:
        encodings.append(np.array(board.pieces(piece_type, chess.WHITE).tolist(), np.int8)
                         - np.array(board.pieces(piece_type, chess.BLACK).tolist(), np.int8))

    return np.concatenate(encodings)


def extract_features_sparse(board):
    encodings = []
    for piece_type in chess.PIECE_TYPES:
        for piece_color in chess.COLORS:
            encodings.append(np.array(board.pieces(piece_type, piece_color).tolist(), dtype=bool))
    return np.concatenate(encodings)


def interpret_training_data(pgn_file):
    pass
