import chess

from engines.naive_bayes.naivebayesengine import NaiveBayesEngine
from engines.deep_learning.deeplearningengine import DeepLearningEngine


def main():
    engine = DeepLearningEngine()  # Change engine here
    engine.load_model()

    # Start game
    b = chess.Board()
    while not b.is_game_over():
        print(b)
        uci_in = input("UCI> ")
        player_move = chess.Move.from_uci(uci_in)
        if player_move in b.legal_moves:
            b.push(player_move)
            if (b.is_game_over()):
                print("You win!")
                break
            engine_move = engine.choose_move(b)
            print("Engine>", engine_move.uci())
            b.push(engine_move)
            if (b.is_game_over()):
                print("You lose!")
                break
        else:
            print("Illegal move.")
    pass


if __name__ == '__main__':
    main()
