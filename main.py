import numpy as np
import chess.pgn


def main():
    # TODO
    print("Hello, world")

    dataset = open("lichess_db_standard_rated_2013-01.pgn")
    game = chess.pgn.read_game(dataset)

    print(game.board())
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(game.next().board())


if __name__ == '__main__':
    main()
