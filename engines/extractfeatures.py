import chess
import chess.pgn
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
            if move_color == winner:  # Only consider winner's moves
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

                print(move_dictionary[key])

            move_color = not move_color

        game_num += 1
        game = chess.pgn.read_game(pgn)

    return move_dictionary

if __name__ == '__main__':
    pgn_file = "../lichess_db_standard_rated_2013-01.pgn"
    interpret_training_data(pgn_file, 100)