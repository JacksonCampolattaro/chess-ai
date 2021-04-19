import logging

# Based on:
# https://github.com/thomasahle/sunfish/blob/master/uci.py

import chess
import re


def main():
    logging.basicConfig(level=logging.DEBUG)

    stack = []
    while True:

        # Get the user's command
        command = stack.pop() if stack else input()

        # If the command is quit, stop parsing
        if command == "quit":
            break

        # If the command is uci, identify yourself
        elif command == "uci":
            print("id name Test")
            print("id author MeanSquares")
            print("uciok")

        # Always report ready when asked
        elif command == "isready":
            print("readyok")

        # Send a fresh board, when asked
        elif command == "ucinewgame":
            print("position fen " + chess.Board().fen())

        # This is how we get told the current board state
        elif command.startswith("position"):

            # Get the different parts of the argument
            arguments = re.split(r"position | moves ", command)[1:]
            starting_position_argument = arguments[0]

            # Parse the starting position
            fen_string = starting_position = starting_position_argument[4:] \
                if starting_position_argument.startswith("fen") else chess.Board().fen()
            logging.debug(starting_position)

            # Parse the moves
            moves_string = "" if len(arguments) == 1 else arguments[1]

            # Parse the moves list
            # TODO
            pass

        # This is how we choose the next move
        elif command.startswith("go"):
            # TODO
            pass

        # Don't do anything for unrecognized commands
        else:
            pass


if __name__ == '__main__':
    main()
