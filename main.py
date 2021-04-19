import numpy as np
import chess.pgn
import chess.engine

import engines.dumb

def main():

    our_engine = engines.dumb.RandomEngine()
    other_engine = chess.engine.SimpleEngine.popen_uci("/usr/bin/stockfish")

    game = chess.pgn.Game()
    board = game.board()
    print(board)

    while not board.is_game_over(claim_draw=True):

        print()

        if board.turn:

            print("Our engine's move:")
            board.push(our_engine.choose_move(board))

        else:

            print("Stockfish's move:")
            result = other_engine.play(board, chess.engine.Limit(time=0.1))
            board.push(result.move)

        print(board)

    print()
    print(board.result())



if __name__ == '__main__':
    main()
