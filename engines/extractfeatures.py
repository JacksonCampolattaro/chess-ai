import chess
import chess.pgn
import numpy as np
import random as rn


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


def create_bitboard(indices: list):
    # Not used
    board = np.empty(shape=64, dtype=bool)
    board[indices] = True
    return board


def create_floatboard(indices: list):
    # Not used
    board = np.zeros(shape=64, dtype=np.float16)
    board[indices] = 1.0
    return board


def interpret_training_data(pgn_file, end_early=-1):
    """
    :param pgn_file:    File name of PGN file to extract data from.
    :param end_early:   Integer to determine after how many games feature extraction should end. If set to -1, function
                        will extract all game data from pgn file.
    :return move_dictionary:
        Move dictionary is python dictionary containing every feature set, i.e. one-hot encoded board state, and label
        set for each of the models. These label sets include: board space to move out of and six arrays for each type
        of piece and into which space it moved.

        move_dictionary Format:
        {key = BoardFEN (string) : value = [[move_space], [pawn_moves], [knight_moves], [bishop_moves], ..., [king_moves]]}
    """
    # Set-up our move_dictionary, i.e. comprehensive feature and label set, and our pgn file
    move_dictionary = {}
    pgn = open(pgn_file)
    game = chess.pgn.read_game(pgn)
    game_num = 1
    while game is not None and (game_num <= end_early or end_early < 0):
        # Get game details and determine a winner
        game_result = game.headers["Result"]
        winner = chess.WHITE * (game_result == "1-0") + chess.BLACK * (game_result == "0-1")
        # Get starting board
        board = game.board()
        # Start playing through the game
        move_color = chess.WHITE
        for move in game.mainline_moves():
            if move_color == winner and move_color == chess.WHITE:  # Only consider winner's moves
                key = board.fen()

                # Add feature set to dictionary if not in there already
                if key not in move_dictionary:
                    move_dictionary[key] = [[], [], [], [], [], [], []]

                # Extract labels from the move
                from_space = move.from_square
                move_piece = board.piece_at(from_space).piece_type
                into_space = move.to_square

                # Place labels into dictionary entry
                move_dictionary[key][0].append(from_space)
                move_dictionary[key][move_piece].append(into_space)

            board.push(move)

            move_color = not move_color

        game_num += 1
        game = chess.pgn.read_game(pgn)

    return move_dictionary


def test_train_split(pgn_file, num_games, percentage_test):
    # Stochastic split of pgn_file into two separate pgn_files
    num_train = int(num_games*(1-percentage_test))
    num_test = int(num_games*percentage_test)
    rng = rn.Random()
    pgn_source = open(pgn_file)
    pgn_train = open("naive_bayes/train.pgn", "w")
    pgn_test = open("naive_bayes/test.pgn", "w")
    while num_train > 0 and num_test > 0:
        train_amt = min(rng.randint(0, 50), num_train)
        test_amt = min(rng.randint(0, int(50*percentage_test)), num_test)
        for i in range(train_amt):
            game = chess.pgn.read_game(pgn_source)
            print(game, file=pgn_train, end="\n\n")
        for i in range(test_amt):
            game = chess.pgn.read_game(pgn_source)
            print(game, file=pgn_test, end="\n\n")

        num_train -= train_amt
        num_test -= test_amt


if __name__ == '__main__':
    pgn_file = "../lichess_db_standard_rated_2013-01.pgn"
    test_train_split(pgn_file, 120000, 0.2)
