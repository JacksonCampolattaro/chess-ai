import naivebayesengine

def main():
    train()
    test()

def train():
    nb = naivebayesengine.NaiveBayesEngine()
    nb.train("lichess_db_standard_rated_2013-01.pgn")
    nb.save_model()

def test():
    nb = naivebayesengine.NaiveBayesEngine()
    nb.load_model()
    scores = nb.test("lichess_db_standard_rated_2013-01.pgn")
    print(scores)


if __name__ == '__main__':
    test()
