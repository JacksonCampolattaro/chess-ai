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