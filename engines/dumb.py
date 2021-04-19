import random

import chess.engine


class RandomEngine:

    def train(self, pgn_file):
        pass

    def choose_move(self, board):
        moves = [move for move in board.legal_moves]
        return random.choice(moves)


class ShallowEngine:

    def train(self, pgn_file):
        pass

    def score(self, board, move):
        board = chess.Board()
        board.pop()

    def choose_move(self, board):
        moves = [move for move in board.legal_moves]
        moves = sorted(moves, key=lambda move: board.push(move))
        return moves[0]
