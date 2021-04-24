#!/usr/bin/env python3

from engine import NaiveBayesEngine

if __name__ == "__main__":
    nb = NaiveBayesEngine()
    nb.load_model()
    nb.main()
