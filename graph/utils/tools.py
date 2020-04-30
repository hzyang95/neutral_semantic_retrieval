import random
import copy

import torch

random.seed(123456)


def shuffle(lol, seed=123456):
    """
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    """
    for l in lol:
        random.seed(seed)
        random.shuffle(l)


if __name__ == "__main__":
    print("------------This is for utility test--------------")
    import numpy as np
    a1 = [1, 2, 3, 4, 5, 6]
    a2 = [7, 8, 9, 10, 11, 12]
    a3 = [a1, a2]
    a4 = np.asarray(a1) + 5
    a5 = np.asarray(a2) + 5
    a6 = [a4, a5]

    shuffle(a3, seed=123456)
    shuffle(a6, seed=123456)
    print(a3)
    print(a6)


def load_torch_model(model_path, use_cuda=True):
    with open(model_path + "/model.pt", "rb") as f:
        if use_cuda:
            model = torch.load(f)
        else:
            model = torch.load(f, map_location=lambda storage, loc: storage)
            model.cpu()
        return model


def pickle_to_data(in_file):
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    with open(in_file, 'rb') as f:
        your_dict = pickle.load(f)
        return your_dict


def data_to_pickle(your_dict, out_file):
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    with open(out_file, 'wb') as f:
        pickle.dump(your_dict, f,protocol = 4)