#!/usr/bin/env python3
import engines.uci
from deeplearningengine import DeepLearningEngine

if __name__ == "__main__":
    dl = DeepLearningEngine()
    dl.load_model()
    engines.uci.main(dl)
