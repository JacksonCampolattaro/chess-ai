import chess.engine
import chess.pgn


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


def grade(engine):
    enemy = chess.engine.SimpleEngine.popen_uci("/usr/bin/stockfish")

    boards = [play(engine, enemy) for _ in range(100)]

    movecounts = [len(board.move_stack) for board in boards]

    print()
    return sum(movecounts) / 100


if __name__ == "__main__":
    our_engine = chess.engine.SimpleEngine.popen_uci("./engines/naive_bayes/naivebayesprogram.py")
    # our_engine = chess.engine.SimpleEngine.popen_uci("/usr/bin/stockfish")
    print(grade(our_engine))

