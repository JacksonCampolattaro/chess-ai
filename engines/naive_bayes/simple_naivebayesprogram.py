#!/usr/bin/env python3
import engines.uci
from simple_naivebayesengine import NaiveBayesEngine

if __name__ == "__main__":
    nb = NaiveBayesEngine()
    nb.load_model()
    engines.uci.main(nb)
