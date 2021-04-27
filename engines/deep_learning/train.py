import engines.deep_learning.deeplearningengine


def main():
    # Create a new engine
    dl = engines.deep_learning.deeplearningengine.DeepLearningEngine()

    # Train it with our data set
    dl.train("lichess_db_standard_rated_2013-01.pgn")

    # Save the generated model
    dl.save_model()


if __name__ == '__main__':
    main()
