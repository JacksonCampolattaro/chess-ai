
class Engine:

    def train(self, data):
        raise NotImplementedError

    def choose_move(self, board):
        raise NotImplementedError

    def save_model(self, file_name):
        raise NotImplementedError

    def load_model(self, file_name):
        raise NotImplementedError

