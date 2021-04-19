import numpy as np
import chess.pgn
import chess.engine

import engines.dumb
import grader


def main():
    our_engine = chess.engine.SimpleEngine.popen_uci("./uci.py")

    print(grader.grade(our_engine))


def play():

    our_engine = chess.engine.SimpleEngine.popen_uci("./uci.py")
    other_engine = chess.engine.SimpleEngine.popen_uci("/usr/bin/stockfish")

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

            print("Stockfish's move:")
            result = other_engine.play(board, chess.engine.Limit(time=0.1))
            board.push(result.move)

        print(board)

    print()
    print(board.result())
    print(len(board.move_stack))


if __name__ == '__main__':
    main()
