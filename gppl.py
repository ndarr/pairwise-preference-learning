import pickle
from sys import path
from glob import glob
from os import getcwd
from os.path import join, realpath, dirname, exists
from sentence_transformers import SentenceTransformer
import csv
from csv import reader
import logging
import numpy as np
from code_gppl.python.models.gp_pref_learning import GPPrefLearning
from metrics import get_alliteration_scores, get_readability_scores, get_rhyme_scores
from utils import POEM_FOLDER, arrange_poem_scores

# this is necessary because gppl code uses absolute imports
__location__ = realpath(join(getcwd(), dirname(__file__)))
path.append(join(__location__, "code_gppl", "python", "models"))

EMBEDDINGS_FILE = join("embeddings", "gppl_embeddings.pkl")
MODEL_FILE = "models/gppl_model_{}.pkl"


def embed_sentences(sentences):
    if exists(EMBEDDINGS_FILE) and False:
        with open(EMBEDDINGS_FILE, 'rb') as f:
            saved_sents, embeddings = pickle.load(f)
            if saved_sents == sentences:
                return embeddings

    model = SentenceTransformer('average_word_embeddings_glove.6B.300d', device="cuda")
    embeddings = np.asarray(model.encode(sentences))
    rhyme_scores = get_rhyme_scores(sentences)
    alliteration_scores = get_alliteration_scores(sentences)
    readability_scores = get_readability_scores(sentences)
    embeddings = np.concatenate((embeddings,
                                 rhyme_scores, alliteration_scores, readability_scores), axis=1)

    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump((sentences, embeddings), f)

    return embeddings


def get_scores_for_cat(start_cat=5, end_cat=-1):
    _, poem_scores = get_accuracy(start_cat, end_cat)
    return poem_scores


def save_scores():
    scores = []
    scores.append(get_accuracy()[1])
    for i in range(5, 15):
        scores.append(get_accuracy(i, i + 1)[1])

    scores_per_poem = arrange_poem_scores(scores)

    print(scores_per_poem)

    header = ['poem', 'all', 'coherent', 'grammatical', 'melodious', 'moved', 'real', 'rhyming',
              'readable', 'comprehensible', 'intense', 'liking']
    with open("scores/gppl_subset.csv", "w+") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
        for poem in scores_per_poem:
            line = [poem.replace("\n", "<br>")] + scores_per_poem[poem]
            csv_writer.writerow(line)


def load_dataset(start_cat=5, end_cat=-1):
    a1_sent_idx = []
    a2_sent_idx = []
    prefs_train = []
    sents = []

    with open("data_poems/multi_annotated_poems_10.txt", "r") as f:
        multi_annotated = [line.strip() for line in f.readlines()]

    for filename in glob(join(POEM_FOLDER, "*.csv")):
        with open(filename, newline='') as f:
            lines = reader(f, dialect="unix")
            categories = next(lines)[start_cat:end_cat]
            for line in lines:
                if line[1] not in multi_annotated and line[2] not in multi_annotated:
                    continue
                line = [col.replace("<br>", "\n").strip() for col in line]
                for text in line[1:3]:
                    if text not in sents:
                        sents.append(text)
                for idx, category in enumerate(categories, start_cat):
                    label = line[idx]

                    if label.strip() == "1":
                        prefs_train.append(1)
                    elif label.strip() == "2":
                        prefs_train.append(-1)
                    elif label.strip():
                        prefs_train.append(0)
                    else:
                        continue

                    a1_sent_idx.append(sents.index(line[1]))
                    a2_sent_idx.append(sents.index(line[2]))
    sent_features = embed_sentences(sents)
    ndims = len(sent_features[0])
    print(len(sents))
    a1_sent_idx = np.array(a1_sent_idx, dtype=int)
    print(a1_sent_idx.shape)
    a2_sent_idx = np.array(a2_sent_idx, dtype=int)
    print(a2_sent_idx.shape)
    prefs_train = np.array(prefs_train, dtype=float)

    return a1_sent_idx, a2_sent_idx, sent_features, prefs_train, ndims, sents


def train_model(filename, optimize=False, start_cat=5, end_cat=-1):
    a1_train, a2_train, sent_features, prefs_train, ndims, _ = load_dataset(start_cat, end_cat)

    shape_s0 = len(a1_train) / 2.0
    rate_s0 = shape_s0 * np.var(prefs_train)
    model = GPPrefLearning(
        ndims, shape_s0=shape_s0, rate_s0=rate_s0)

    model.max_iter = 2000

    model.fit(a1_train, a2_train, sent_features, prefs_train,
              optimize=optimize, use_median_ls=True, input_type='zero-centered')
    with open(filename, 'wb') as fh:
        pickle.dump(model, fh)
    return model


def compute_scores():
    if not exists(MODEL_FILE):
        print("GPPL: no trained model found")
        return 0
    with open(MODEL_FILE, 'rb') as fh:
        model = pickle.load(fh)
        return model.predict_f()[0]


def get_accuracy(start_cat=5, end_cat=-1):
    a1_train, a2_train, sent_features, prefs_train, ndims, sents = load_dataset(start_cat, end_cat)
    if start_cat == 5 and end_cat == -1:
        model_id = "all"
    else:
        model_id = str(start_cat)

    if not exists(MODEL_FILE.format(model_id)):
        print("GPPL: no trained model found")
        return 0
    with open(MODEL_FILE.format(model_id), 'rb') as fh:
        model = pickle.load(fh)
        scores = model.predict_f()[0]
    correct = 0.
    wrong = 0.
    poem_scores = {}
    for idx in range(len(a1_train)):
        poem1_score = scores[a1_train[idx]][0]
        poem2_score = scores[a2_train[idx]][0]
        if poem1_score > poem2_score and prefs_train[idx] == 1:
            correct += 1
        elif poem2_score > poem1_score and prefs_train[idx] == -1:
            correct += 1
        elif poem1_score == poem2_score:
            continue
        else:
            wrong += 1

        poem1 = sents[a1_train[idx]]
        poem2 = sents[a2_train[idx]]
        if poem1 not in poem_scores:
            poem_scores[poem1] = poem1_score
        if poem2 not in poem_scores:
            poem_scores[poem2] = poem2_score
    return correct / (correct + wrong), poem_scores


def get_all_accuracies():
    # Get Main accuracy accross all categories
    print(f"GPPL acc all: {get_accuracy()[0]}")
    for i in range(5, 15):
        print(f"GPPL acc {i}: {get_accuracy(start_cat=i, end_cat=i + 1)[0]}")


if __name__ == "__main__":
    # Train on all categories
    train_model(MODEL_FILE.format("all"), optimize=True)
    # Train categories independently
    for i in range(5, 15):
        get_all_accuracies()
        train_model(MODEL_FILE.format(i), optimize=True, start_cat=i, end_cat=i + 1)
    save_scores()
    get_all_accuracies()
