import chess.pgn


def choose_move(board):
    # TODO This is an example of what our trained model will do!

    # Get the list of legal moves
    legal_moves = board.legal_moves

    # Pick one, and return it
    return legal_moves[0]
