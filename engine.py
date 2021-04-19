import logging


# Based on:
# https://github.com/thomasahle/sunfish/blob/master/uci.py

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
        if command == "uci":
            print("id name Test")
            print("id author MeanSquares")
            print("uciok")

        logging.debug(command)


if __name__ == '__main__':
    main()
