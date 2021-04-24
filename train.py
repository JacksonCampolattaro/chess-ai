import engines.old_naive_bayes.naivebayesengine


def main():
    dl = engines.deep_learning.deeplearningengine.DeepLearningEngine()
    dl.train("/home/jackcamp/Documents/chess-ai/lichess_db_standard_rated_2013-01.pgn")


if __name__ == '__main__':
    main()
