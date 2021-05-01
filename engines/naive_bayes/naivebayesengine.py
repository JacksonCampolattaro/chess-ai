"""
This approach was developed in response to the old Naive-Bayes implementation's lackluster performance and it's lack of
'awareness.' So, we adapted
"""

import os.path

import numpy as np
import chess
from engines import extractfeatures
from engines.engine import Engine


class NaiveBayesEngine(Engine):

    def __init__(self):
        self.move_select_model = []

        self.pawn_model = []
        self.knight_model = []
        self.bishop_model = []
        self.rook_model = []
        self.queen_model = []
        self.king_model = []

        self.last_move = None
        self.backtrack_penalty = 0.1

    def train(self, pgn_file, train_limit=-1):
        """
        Counts the occurrence of each moves and the conditional occurrence of each feature to generate the Naive Bayes
        model's internal probability matrices.

        :param pgn_file: String of the .pgn file that the model will train on.
        :param train_limit: Integer that defines the maximum number of games the model will train on. Default value is
                -1, which allows the model to train on the .pgn file's entire dataset.
        """
        # Extract data from pgn file using extractfeatures.py
        move_dict = extractfeatures.interpret_training_data(pgn_file, train_limit)
        # Set up counting vectors
        labelCount = np.zeros([7, 64], dtype=int)
        conditionalFeatureCount = np.zeros([7, 64, 768], dtype=int)
        # Go through each entry in the dictionary and add respective values into count vectors
        for board_key in move_dict:
            board = chess.Board(board_key)
            feature_set = extractfeatures.extract_features_sparse(board)
            fs_indexes = np.argwhere(feature_set == 1)
            for i in range(7):
                for label in move_dict[board_key][i]:
                    labelCount[i, label] += 1
                    conditionalFeatureCount[i, label, fs_indexes] += 1

        prior_prob = [row / np.sum(row) for row in labelCount]
        div = np.repeat(np.sum(conditionalFeatureCount, axis=2)[:, :, np.newaxis], 768, axis=2) + 768
        with np.errstate(invalid='ignore', divide='ignore'):
            conditional_prob = (conditionalFeatureCount + 1) / div

        # Store data into the models
        self.move_select_model = self.prep_model(prior_prob[0], conditional_prob[0])
        self.pawn_model = self.prep_model(prior_prob[chess.PAWN], conditional_prob[chess.PAWN])
        self.knight_model = self.prep_model(prior_prob[chess.KNIGHT], conditional_prob[chess.KNIGHT])
        self.bishop_model = self.prep_model(prior_prob[chess.BISHOP], conditional_prob[chess.BISHOP])
        self.rook_model = self.prep_model(prior_prob[chess.ROOK], conditional_prob[chess.ROOK])
        self.queen_model = self.prep_model(prior_prob[chess.QUEEN], conditional_prob[chess.QUEEN])
        self.king_model = self.prep_model(prior_prob[chess.KING], conditional_prob[chess.KING])

    def prep_model(self, prior_prob, conditional_prob):
        """
        Places given prior probability and conditional probability vectors/matrices into a model matrix.

        Matrix format (64x769):
        P(y = A1)                   P(y = B1)                  P(y = C1)                  ...  P(y = H8)
        P(w_pawn @ A1 | y = A1)     P(w_pawn @ A1 | y = B1)    P(w_pawn @ A1 | y = C1)    ...  P(w_pawn @ A1 | y = H8)
        P(w_pawn @ B1 | y = A1)     P(w_pawn @ B1 | y = B1)    P(w_pawn @ B1 | y = C1)    ...  P(w_pawn @ B1 | y = H8)
        ...
        P(b_pawn @ A1 | y = A1)     P(b_pawn @ A1 | y = B1)    P(b_pawn @ A1 | y = C1)    ...  P(b_pawn @ A1 | y = H8)
        ...
        P(w_knight @ A1 | y = A1)   P(w_knight @ A1 | y = B1)  P(w_knight @ A1 | y = C1)  ...  P(w_knight @ A1 | y = H8)
        ...
        P(b_king @ H8 | y = A1)     P(b_king @ H8 | y = B1)    P(b_king @ H8 | y = C1)    ...    P(b_king @ H8 | y = H8)

        :param prior_prob: 64-element vector of prior probabilities of the labels
        :param conditional_prob: 64x768 matrix of conditional probabilities of features given labels
        :return: 64x769 model matrix
        """
        model = np.zeros([64, 768 + 1])
        model[:, 0] = prior_prob
        model[:, 1:] = conditional_prob
        return model

    def get_scores(self, feature_set, model):
        """
        Calculates the 64-element probability density matrix (i.e. the "scores" of each chess square) given a certain
        model matrix and feature_set using Naive Bayes.

        :param feature_set:
        :param model:
        :return:
        """
        where_inv = np.argwhere(feature_set == 0) + 1
        updated_model = model.copy()
        updated_model[:, where_inv] = 1 - model[:, where_inv]
        scores = np.prod(updated_model[:, 1:], axis=1)
        scores = scores * updated_model[:, 0]
        return scores

    def choose_move(self, board):
        """
        Predicts a legal, optimal move given the board feature set and trained model.

        :param board: python-chess.Board, current board state used to get the model's prediction
        :return: python-chess.Move, model's prediction based on inputted board and internal probabilities
        """
        # First, select which piece should be moved
        feature_set = extractfeatures.extract_features_sparse(board)
        # Get all possible moves and their starting locations
        possible_moves = np.array(list(board.legal_moves))
        possible_from_squares = np.unique([move.from_square for move in possible_moves])
        # Use NaiveBayes move_select_model to select the best starting (from) square
        scores = self.get_scores(feature_set, self.move_select_model)
        mask = np.zeros(scores.shape)
        mask[possible_from_squares] = 1
        scores = scores * mask  # Set all illegal from squares to 0
        best_from_square = np.argmax(scores)
        # Create possible ending (to) squares based on selection
        possible_moves = [move for move in possible_moves if move.from_square == best_from_square]
        possible_to_squares = np.unique(
            [move.to_square for move in possible_moves if move.from_square == best_from_square])
        # Select movement model based on piece type
        move_piece = board.piece_at(best_from_square).piece_type
        piece_model = self.select_model(move_piece)
        # Use selected NaiveBayes model to select the best to square
        scores = self.get_scores(feature_set, piece_model)
        mask = np.zeros(scores.shape)
        mask[possible_to_squares] = 1
        scores = scores * mask  # Set all illegal to squares to 0
        # Penalize backtracking
        if self.last_move is not None and best_from_square == self.last_move.to_square:
            scores[self.last_move.from_square] *= self.backtrack_penalty
        best_to_square = np.argmax(scores)
        # Create move based on findings
        move = chess.Move(from_square=best_from_square, to_square=best_to_square)
        self.last_move = move
        return move

    def select_model(self, piece_type):
        """
        Abstract dictionary that returns a class model based on a python-chess.piece_type

        :param piece_type: Type of chess piece defined in python-chess library.
        :return: 64x769 element class sub-model probability matrix.
        """
        if piece_type == chess.PAWN:
            return self.pawn_model
        elif piece_type == chess.KNIGHT:
            return self.knight_model
        elif piece_type == chess.BISHOP:
            return self.bishop_model
        elif piece_type == chess.ROOK:
            return self.rook_model
        elif piece_type == chess.QUEEN:
            return self.queen_model
        elif piece_type == chess.KING:
            return self.king_model
        else:
            return self.move_select_model

    def has_model(self, file_name="naive_bayes_model"):
        """
        Returns whether a pre-trained and saved model exists.

        :param file_name: Name of pre-trained model. Default: "naive_bayes_model"
        :return: Boolean value indicating whether the file exists.
        """
        directory = os.path.dirname(os.path.realpath(__file__))
        return os.path.exists(os.path.join(directory, file_name) + ".npy")

    def save_model(self, file_name="naive_bayes_model"):
        """
        Saves the Naive Bayes object's models into a numpy file, so that it may be reloaded later on.

        :param file_name: Name of file to write to. Default: "naive_bayes_model"
        """
        models = np.array([self.move_select_model, self.pawn_model, self.knight_model,
                           self.bishop_model, self.rook_model, self.queen_model, self.king_model])
        directory = os.path.dirname(os.path.realpath(__file__))
        np.save(os.path.join(directory, file_name), models)

    def load_model(self, file_name="naive_bayes_model"):
        """
        Reads a pre-trained model and saves it to the object.

        :param file_name: Name of pre-trained model. Default: "naive_bayes_model"
        """
        directory = os.path.dirname(os.path.realpath(__file__))
        models = np.load(os.path.join(directory, file_name) + ".npy", allow_pickle=True)
        self.move_select_model = models[0]
        self.pawn_model = models[chess.PAWN]
        self.knight_model = models[chess.KNIGHT]
        self.bishop_model = models[chess.BISHOP]
        self.rook_model = models[chess.ROOK]
        self.queen_model = models[chess.QUEEN]
        self.king_model = models[chess.KING]

    def test(self, pgn_file, test_limit=-1):
        correct_count = np.zeros(7)
        total_count = np.zeros(7)
        pgn = open(pgn_file)
        game = chess.pgn.read_game(pgn)
        game_num = 1
        while game is not None and (game_num <= test_limit or test_limit < 0):

            # Get starting board
            board = game.board()
            # Start playing through the game
            for actual_move in game.mainline_moves():
                total_count[0] += 1
                predicted_move = self.choose_move(board)
                predicted_piece = board.piece_at(predicted_move.from_square).piece_type
                correct_count[0] += (predicted_move.from_square == actual_move.from_square)
                total_count[predicted_piece] += (predicted_move.from_square == actual_move.from_square)
                correct_count[predicted_piece] += ((predicted_move.from_square == actual_move.from_square) &
                                                   (predicted_move.to_square == actual_move.to_square))
                scores = correct_count / total_count
                print(scores)
                board.push(actual_move)

            game_num += 1
            game = chess.pgn.read_game(pgn)

        scores = correct_count / total_count
        return scores
