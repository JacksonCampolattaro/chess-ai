import chess.engine
import chess.pgn
import engines.naive_bayes.naivebayesengine
import engines.deep_learning.deeplearningengine


def play(engine1, engine2):
    game = chess.pgn.Game()
    board = game.board()

    while not board.is_game_over(claim_draw=True):

        if board.turn:

            result = engine1.play(board, chess.engine.Limit(time=0.1))
            board.push(result.move)

        else:

            result = engine2.play(board, chess.engine.Limit(time=0.1))
            board.push(result.move)

    print("W" if board.outcome(claim_draw=True).winner else "L", end="")

    return board

def simple_play(print_board = False):
    engine_white = engines.naive_bayes.naivebayesengine.NaiveBayesEngine()
    engine_black = engines.deep_learning.deeplearningengine.DeepLearningEngine()
    engine_white.load_model()
    engine_black.load_model()

    turn = chess.WHITE
    board = chess.Board()
    if print_board:
        print(board)
        print()

    moves = []
    while not board.is_game_over(claim_draw=True):
        if turn == chess.WHITE:
            move = engine_white.choose_move(board)
        else:
            move = engine_black.choose_move(board)

        moves.append(move)
        board.push(move)

        if print_board:
            print(("White"*turn)+("Black"*(not turn)), move.uci())
            print(board)
            print()

        turn = not turn

    return moves




def grade(engine):
    enemy = chess.engine.SimpleEngine.popen_uci("./engines/deep_learning/deeplearningprogram.py")

    boards = [play(engine, enemy) for _ in range(100)]

    movecounts = [len(board.move_stack) for board in boards]

    print()
    return sum(movecounts) / 100


if __name__ == "__main__":
    #our_engine = chess.engine.SimpleEngine.popen_uci("./engines/naive_bayes/naivebayesprogram.py")
    # our_engine = chess.engine.SimpleEngine.popen_uci("/usr/bin/stockfish")
    #print(grade(our_engine))
    moves = simple_play(True)
    print(moves)
