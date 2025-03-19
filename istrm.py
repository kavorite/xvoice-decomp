from os import scandir
import numpy as np

np_files = [ent.path for ent in scandir("tokens")]

def load_tokens(np_files, chunk_size):
    buffer = []
    seq_ids = []
    labels = []
    length = 0
    for seq_id, file in enumerate(np_files):
        tokens = np.load(file)
        seq_ids.append(seq_id)
        buffer.append(tokens)
        length += tokens.size
        if length > chunk_size:
            tids = np.concatenate(buffer)
            sids = np.repeat(seq_ids, repeats=[len(a) for a in buffer])
            while length >= chunk_size:
                yield tids[:chunk_size], sids[:chunk_size]
                tids, sids = tids[chunk_size:], sids[chunk_size:]
                length = tids.size
            buffer = [tids]
            seq_ids = [sids[-1]]
