# chess-ai
A simple chess AI in Python,
using datasets from [Lichess](https://database.lichess.org/)

## Structure

We produced several models during our development process,
all chess engines are contained in the `engines` directory.

The relevant engines for this project include `naive_bayes` and `deep_learning`.
Other engines include `old_naive_bayes` which chooses the most common move without looking
at the current state of the board,
and `random_engine` which chooses randomly from the pool of legal moves.

Each engine provides a class (e.g. `NaiveBayesEngine`) which implements the `Engine` interface 
defined in `engine.py`. The classes provide the following functionality:
* `engine.train(filepath)` Load a dataset of games,
  and use the data to train the model.
* `engine.choose_move(board)` Returns the model's choice of move, 
  based on the current board state. 
* `engine.save_model(filename)` Saves the current (trained) model in a file 
  adjacent to the engine's file.
* `engine.load_model(filename)` Loads a previously saved model from a file
  adjacent to the engine's file.

For each engine, there is also a `train.py`, running this 
invokes the engine's train method and saves the new model.

We also provide `*program.py` for each engine.
Running this starts an engine which speaks the UCI standard,
this allows us to use the engines with different interfaces.
`uci.py` is the adapter which enables the use of different `Engine` objects
with the UCI standard.

## Usage

To try out one of the engines, we recommend running `uci_game.py`
with python 3.8. (With a modern IDE, dependencies should be automatically read
from `requirements.txt` and installed in your `venv`)

`uci_game.py` provides an ascii chess board with which you can play against
an engine that conforms to our interface. 
Engines can be swapped by changing the first line of `main()` to load
a different engine.

Play is done by sending commands in the UCI format,
for example, "e2e3" moves the pawn at column E forward by 1 square.


