# chess-ai
A simple chess AI in Python,
using datasets from [Lichess](https://database.lichess.org/)

## Usage

We produced several models during our development process,
all chess engines are contained in the `engines` directory.

The relevant engines for this project include `naive_bayes` and `deep_learning`.

Each engine provides a class (e.g. `NaiveBayesEngine`) with a specific interface.
These have a standard interface, with the following functionality:
* `engine.train(filepath)` Load a dataset of games,
  and use the data to train the model.
* `engine.choose_move(board)` Returns the model's choice of move, 
  based on the current board state. 
* `engine.save_model(filename)` Saves the current (trained) model in a file 
  adjacent to the engine's file.
* `engine.load_model(filename)` Loads a previously saved model from a file
  adjacent to the engine's file.


### Naive Bayes

#### Training

#### Running

### Deep Learning

#### Training

#### Running

