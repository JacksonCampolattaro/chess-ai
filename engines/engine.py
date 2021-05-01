"""
engine.py || Abstract Engine Interface

Establishes the basic interface of all engines within the project.
"""
class Engine:

    def train(self, data):
        raise NotImplementedError

    def choose_move(self, board):
        raise NotImplementedError

    def save_model(self, file_name):
        raise NotImplementedError

    def load_model(self, file_name):
        raise NotImplementedError

