import numpy
import engines.uci


class RandomEngine(engines.uci.UCIEngine):

    def train(self, data):
        pass

    def choose_move(self, board):
        moves = [move for move in board.legal_moves]
        return numpy.random.choice(moves)

    def save_model(self, file_name):
        pass

    def load_model(self, file_name):
        pass
