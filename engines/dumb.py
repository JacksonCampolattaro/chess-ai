import random

import chess.engine


class RandomEngine:

    def choose_move(self, board):
        moves = [move for move in board.legal_moves]
        return random.choice(moves)


class ShallowEngine:

    def score(self, board, move):
        board = chess.Board()
        board.pop()

    def choose_move(self, board):
        moves = [move for move in board.legal_moves]
        moves = sorted(moves, key=lambda move: board.push(move))
        return moves[0]
