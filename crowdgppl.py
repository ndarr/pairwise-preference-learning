import pickle
from sys import path
from glob import glob
from os import getcwd
from os.path import join, realpath, dirname, exists
from sentence_transformers import SentenceTransformer
from csv import reader
import csv
import logging
import numpy as np
from code_gppl.python.models.collab_pref_learning_svi import CollabPrefLearningSVI
from metrics import get_alliteration_scores, get_readability_scores, get_rhyme_scores
from utils import POEM_FOLDER, arrange_poem_scores

# this is necessary because gppl code uses absolute imports
__location__ = realpath(join(getcwd(), dirname(__file__)))
path.append(join(__location__, "code_gppl", "python", "models"))


EMBEDDINGS_FILE = join("embeddings", "embeddings.pkl")
MODEL_FILE = join("models", "crowdgppl_model.pkl")


def embed_sentences(sentences, fresh=False):
    if exists(EMBEDDINGS_FILE) and not fresh:
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
        scores.append(get_accuracy(i, i+1)[1])

    scores_per_poem = arrange_poem_scores(scores)

    print(scores_per_poem)

    header = ['poem', 'all', 'coherent', 'grammatical', 'melodious', 'moved', 'real', 'rhyming',
              'readable', 'comprehensible', 'intense', 'liking']
    with open("scores/crowdgppl_subset.csv", "w+") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
        for poem in scores_per_poem:
            line = [poem.replace("\n", "<br>")] + scores_per_poem[poem]
            csv_writer.writerow(line)

def load_dataset(start_cat=5, end_cat=-1):
    person_train = []
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
            # categories = next(lines)[3:]
            for line in lines:
                if line[1] not in multi_annotated and line[2] not in multi_annotated:
                    continue
                line = [col.replace("<br>", "\n").strip() for col in line]
                for text in line[1:3]:
                    if text not in sents:
                        sents.append(text.replace("<br>", "\n"))
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

                    person_train.append(category)
                    a1_sent_idx.append(sents.index(line[1].replace("<br>", "\n")))
                    a2_sent_idx.append(sents.index(line[2].replace("<br>", "\n")))
    sent_features = embed_sentences(sents)

    ndims = len(sent_features[0])

    id2idx = dict([(v, k) for k, v in dict(
        enumerate(np.unique(person_train))).items()])

    person_train_idx = np.array([id2idx[id_] for id_ in person_train], dtype=int)
    a1_sent_idx = np.array(a1_sent_idx, dtype=int)
    a2_sent_idx = np.array(a2_sent_idx, dtype=int)
    prefs_train = np.array(prefs_train, dtype=float)
    print(len(sents))
    return person_train_idx, a1_sent_idx, a2_sent_idx, sent_features, prefs_train, ndims, sents


def train_model(filename, start_cat=5, end_cat=-1):
    # Train a model...
    person_train_idx, a1_train, a2_train, sent_features, prefs_train, ndims, _ = load_dataset(start_cat, end_cat)

    model = CollabPrefLearningSVI(ndims, shape_s0=2, rate_s0=200, use_lb=True,
            use_common_mean_t=True, ls=None)

    model.max_iter = 2000

    model.fit(person_train_idx, a1_train, a2_train, sent_features, prefs_train, optimize=False,
              use_median_ls=True, input_type='zero-centered')

    logging.info("**** Completed training GPPL ****")

    # Save the model in case we need to reload it

    with open(filename, 'wb') as fh:
        pickle.dump(model, fh)
    del model


def compute_scores():
    if not exists(MODEL_FILE):
        print("CrowdGPPL: no trained model found")
        return 0
    with open(MODEL_FILE, 'rb') as fh:
        model = pickle.load(fh)
        return model.predict_t()

def get_accuracy(start_cat=5, end_cat=-1):
    person_train_idx, a1_train, a2_train, sent_features, prefs_train, ndims, sents = load_dataset(start_cat, end_cat)

    if not exists(MODEL_FILE + str(start_cat)):
        print("GPPL: no trained model found")
        return 0
    with open(MODEL_FILE + str(start_cat), 'rb') as fh:
        model = pickle.load(fh)
        scores = model.predict_f()
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
    return correct/(correct+wrong), poem_scores

def print_all_accuracies():
    # Get accuracy accross all categories
    print(f"CrowdGPPL acc all: {get_accuracy()[0]}")
    for i in range(5, 15):
        print(f"CrowdGPPL acc {i}: {get_accuracy(start_cat=i, end_cat=i+1)[0]}")



if __name__ == "__main__":
    # Train all categories together
    train_model(filename=MODEL_FILE)
    # Train categories independently
    for i in range(5, 15):
        train_model(MODEL_FILE+str(i), start_cat=i, end_cat=i+1)
    print_all_accuracies()
    save_scores()
