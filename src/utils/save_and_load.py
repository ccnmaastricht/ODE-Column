import tomllib
import pickle



def load_config(fn):
    """ Load and return configuration from TOML file. """
    with open(fn, 'rb') as f:
        return tomllib.load(f)

def load_pkl_file(fn):
    """ Load the data from a pickle file. """
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl_file(fn, data):
    """ Save the data to a pickle file. """
    with open(fn, 'wb') as  f:
        pickle.dump(data, f)

