import engines.chessAI_nb
import engines.dumb
import chess


def main():
    engine = engines.chessAI_nb.ChessAI_Naive_Bayes_Engine() # Change engine here
    train_file = "lichess_db_standard_rated_2013-01.pgn"

    engine.train(train_file)

    # Start game
    b = chess.Board()
    while not b.is_game_over():
        print(b)
        uci_in = input("UCI> ")
        player_move = chess.Move.from_uci(uci_in)
        if player_move in b.legal_moves:
            b.push(player_move)
            nb_move = engine.choose_move(b)
            print("NB>", nb_move.uci())
            b.push(nb_move)
        else:
            print("Illegal move.")
    pass



if __name__ == '__main__':
    main()
