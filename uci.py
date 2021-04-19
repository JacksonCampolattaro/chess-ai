#!/usr/bin/env python3

import logging

# Based on:
# https://github.com/thomasahle/sunfish/blob/master/uci.py

import chess
import re

import engines.dumb


def main():
    logging.basicConfig(level=logging.CRITICAL)

    board = chess.Board()
    engine = engines.dumb.RandomEngine()

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
            pass
            #print("position fen " + chess.Board().fen())

        # This is how we get told the current board state
        elif command.startswith("position"):

            # Get the different parts of the argument
            arguments = re.split(r"position | moves ", command)[1:]
            starting_position_argument = arguments[0]

            # Parse the starting position
            board = chess.Board(starting_position_argument[4:]) \
                if starting_position_argument.startswith("fen") else chess.Board()

            # Parse the moves
            moves = [] if len(arguments) == 1 else arguments[1].split(' ')

            # Apply each move to the board
            for move_string in moves:
                board.push(chess.Move.from_uci(move_string))

        # This is how we choose the next move
        elif command.startswith("go"):

            print("bestmove " + engine.choose_move(board).uci())

        # Don't do anything for unrecognized commands
        else:
            pass


if __name__ == '__main__':
    main()
