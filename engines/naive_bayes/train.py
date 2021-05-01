"""
train.py || Naive-Bayes Train Script

Simple script to demonstrate Naive-Bayes engine. Currently configured to perform an 80/20 train-test split and
evaluate performance of the model.
"""
import naivebayesengine
from engines.naive_bayes import extractfeatures
import os

dataset = "lichess_db_standard_rated_2013-01.pgn"


def main():
    # Configured to do a test_train split
    if not os.path.exists("train.pgn") or not os.path.exists("test.pgn"):
        extractfeatures.test_train_split()
    train("train.pgn")
    test("train.pgn")
    test("test.pgn")


def train(pgn_file=dataset):
    # Constructs a Naive-Bayes model, trains it on the .pgn file, then saves the created probability matrices for
    # later use.
    nb = naivebayesengine.NaiveBayesEngine()
    nb.train(pgn_file)
    nb.save_model()


def test(pgn_file=dataset):
    # Constructs an NB model, loads the saved probability matrices (created by train()) then evaluates the model's
    # performance on the given .pgn file.
    nb = naivebayesengine.NaiveBayesEngine()
    nb.load_model()
    scores = nb.test(pgn_file)
    print("NB Model's Accuracy on " + pgn_file + ":")
    print("\"From Sqaure\" Predictor:", scores[0])
    print("PAWN \"To Sqaure\" Predictor:", scores[1])
    print("KNIGHT \"To Sqaure\" Predictor:", scores[2])
    print("BISHOP \"To Sqaure\" Predictor:", scores[3])
    print("ROOK \"To Sqaure\" Predictor:", scores[4])
    print("QUEEN \"To Sqaure\" Predictor:", scores[5])
    print("KING \"To Sqaure\" Predictor:", scores[6], end="\n\n")


if __name__ == '__main__':
    main()
