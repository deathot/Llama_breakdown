import os
import pickle
import requests
import numpy as np

# download tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)
        
with open(input_file_path, 'r') as f:
    data = f.read()
print(f"len(dataset in chars): {len(data):,}")

# get all the unique char that occur in the text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique chars:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from chars to integers
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # take a string, output a list if intergers
def decode(l):
    return ''.join([itos[i] for i in l]) # take list of int, output a str

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9)]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta infor as well, to encode/decode 
meta = {
    'vacao_size': vocab_size,
    'itos': itos,
    'stoi': stoi

}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)