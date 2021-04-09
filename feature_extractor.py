"""
feature_extractor.py

Runs through Lichess dataset and extracts mode label (i.e. most common move) for each feature (board state).

% Record moves in data
for game in data
    board = starting board
    for move in game:
        move_taken = move
        record([board, move_taken])
        board = board+move

for board in boards
    best_move = mode(moves)
    record(best_move)

"""
import chess.pgn

prune_loser = False  # [TODO] Make an option to only consider winners' moves.
end_early = True
end_amount = 100

def extract_boards_and_moves(pgn_file):
    move_dictionary = {}
    pgn = open(pgn_file)
    game = chess.pgn.read_game(pgn)
    game_num = 1
    while game is not None and (game_num <= end_amount or not end_early):
        print("Game #{}".format(game_num))
        board = game.board()
        for move in game.mainline_moves():
            if board.fen() not in move_dictionary:
                move_dictionary[board.fen()] = []
            move_dictionary[board.fen()].append(move)
            board.push(move)
        game = chess.pgn.read_game(pgn)
        game_num += 1
    return move_dictionary

def get_labels_from_dict(move_dict, feature_filename, label_filename):
    feature_file = open(feature_filename, 'w')
    label_file = open(label_filename, 'w')
    for board in move_dict:
        feature_file.write(board + "\n")
        moves = move_dict[board]
        mode_move = max(set(moves), key=moves.count)
        label_file.write(mode_move.__str__() + "\n")


if __name__ == '__main__':
    pgn_file = "lichess_db_standard_rated_2013-01.pgn"
    move_dictionary = extract_boards_and_moves(pgn_file)
    get_labels_from_dict(move_dictionary, "features.txt", "labels.txt")
