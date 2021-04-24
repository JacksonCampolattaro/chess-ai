import engines.deep_learning.deeplearningengine
import engines.naive_bayes.simple_naivebayesengine


def main():
    # dl = engines.deep_learning.deeplearningengine.DeepLearningEngine()
    # dl.train("/home/jackcamp/Documents/chess-ai/lichess_db_standard_rated_2013-01.pgn")
    nb = engines.naive_bayes.simple_naivebayesengine.NaiveBayesEngine()
    nb.train("/home/jackcamp/Documents/chess-ai/lichess_db_standard_rated_2013-01.pgn")
    nb.save_model()


if __name__ == '__main__':
    main()
