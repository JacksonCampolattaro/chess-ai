import chess.engine
import chess.pgn


def play(engine1, engine2):
    game = chess.pgn.Game()
    board = game.board()

    print(".", end="")

    while not board.is_game_over(claim_draw=True):

        if board.turn:

            result = engine1.play(board, chess.engine.Limit(time=0.1))
            board.push(result.move)

        else:

            result = engine2.play(board, chess.engine.Limit(time=0.1))
            board.push(result.move)

    return board


def grade(engine):
    enemy = chess.engine.SimpleEngine.popen_uci("/usr/bin/stockfish")

    boards = [play(engine, enemy) for _ in range(100)]

    movecounts = [len(board.move_stack) for board in boards]

    return sum(movecounts) / 100



