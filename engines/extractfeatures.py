import chess
import numpy as np


def extract_features(board):
    # Create a numpy boolean array
    encodings = []
    for piece_type in chess.PIECE_TYPES:
        encodings.append(np.array(board.pieces(piece_type, chess.WHITE).tolist(), np.int8)
                         - np.array(board.pieces(piece_type, chess.BLACK).tolist(), np.int8))

    return np.concatenate(encodings)
