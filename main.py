import chess.pgn
import chess.engine
import engines.naive_bayes.naivebayesengine

import grader


def main():
    nb = engines.naive_bayes.naivebayesengine.NaiveBayesEngine()
    nb.train("/home/jackcamp/Documents/chess-ai/lichess_db_standard_rated_2013-01.pgn")
    nb.save_model()


def play():
    # our_engine = chess.engine.SimpleEngine.popen_uci("./engines/random/naivebayesprogram.py")
    # other_engine = chess.engine.SimpleEngine.popen_uci("/usr/bin/stockfish")

    our_engine = chess.engine.SimpleEngine.popen_uci("./engines/naive_bayes/naivebayesprogram.py")
    other_engine = chess.engine.SimpleEngine.popen_uci("./engines/random/randomprogram.py")

    game = chess.pgn.Game()
    board = game.board()
    print(board)

    while not board.is_game_over(claim_draw=True):

        print()

        if board.turn:

            print("Our engine's move:")
            result = our_engine.play(board, chess.engine.Limit(time=1000))
            board.push(result.move)

        else:

            print("Other engine's move:")
            result = other_engine.play(board, chess.engine.Limit(time=0.1))
            board.push(result.move)

        print(board)

    print()
    print(board.result())
    print(len(board.move_stack))


if __name__ == '__main__':
    play()
