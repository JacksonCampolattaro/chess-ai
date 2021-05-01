import naivebayesengine
from engines import extractfeatures
import os


def main():
    # Configured to do a test_train split
    if not os.path.exists("train.pgn") or not os.path.exists("test.pgn"):
        extractfeatures.test_train_split("lichess_db_standard_rated_2013-01.pgn")
    train("train.pgn")
    test("train.pgn")
    test("test.pgn")


def train(pgn_file="lichess_db_standard_rated_2013-01.pgn"):
    nb = naivebayesengine.NaiveBayesEngine()
    nb.train(pgn_file)
    nb.save_model()


def test(pgn_file="lichess_db_standard_rated_2013-01.pgn"):
    nb = naivebayesengine.NaiveBayesEngine()
    nb.load_model()
    scores = nb.test(pgn_file)
    print(pgn_file + " scores:")
    print(scores)


if __name__ == '__main__':
    main()
