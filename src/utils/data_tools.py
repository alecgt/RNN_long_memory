import numpy as np

# helper functions for embedding text data
def load_glove_embeddings(gloveFile):  # load GloVe file as dict
    print("Loading GloVe embeddings from {}".format(gloveFile))
    f = open(gloveFile, 'r')
    model = {}
    for line in f:
        splitline = line.split()
        word = splitline[0]
        embedding = np.array([float(val) for val in splitline[1:]])
        model[word] = embedding
    print("Finished. Dict contains {} words!".format(len(model)))
    return model


def embed_text(full_text, embed):  # embed text
    print('Embedding text...')
    embed_dim = len(next(iter(embed.values())))
    text_emb = np.zeros((embed_dim, len(full_text)))
    for t in range(len(full_text)):
        text_emb[:, t] = embed.get(full_text[t].lower(), embed['unk'])
    print('Finished. {} words embedded!'.format(t+1))
    return text_emb
