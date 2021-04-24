import engines.old_naive_bayes.naivebayesengine


def main():
    nb = engines.naive_bayes.naivebayesengine.NaiveBayesEngine()
    nb.train("/home/jackcamp/Documents/chess-ai/lichess_db_standard_rated_2013-01.pgn")
    nb.save_model()


if __name__ == '__main__':
    main()
