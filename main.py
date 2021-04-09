import numpy as np
import chess.pgn


def main():

    dataset = open("lichess_db_standard_rated_2013-01.pgn")
    game = chess.pgn.read_game(dataset)

    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
        print(board)
        print("~~~~~~~~~~~~~~~")


if __name__ == '__main__':
    main()
