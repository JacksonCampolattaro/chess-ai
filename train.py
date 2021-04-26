import engines.deep_learning.deeplearningengine
import engines.old_naive_bayes.old_naivebayesengine
import engines.naive_bayes.naivebayesengine

def main():
    # dl = engines.deep_learning.deeplearningengine.DeepLearningEngine()
    # dl.train("/home/jackcamp/Documents/chess-ai/lichess_db_standard_rated_2013-01.pgn")
    nb = engines.naive_bayes.naivebayesengine.NaiveBayesEngine()
    nb.train("lichess_db_standard_rated_2013-01.pgn")
    nb.save_model()


if __name__ == '__main__':
    main()
