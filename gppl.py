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

__location__ = realpath(join(getcwd(), dirname(__file__)))
path.append(join(__location__, "code_gppl", "python", "models"))

from code_gppl.python.models.gp_pref_learning import GPPrefLearning
from utils import POEM_FOLDER, write_scores_to_file, format_model_filename


EMBEDDINGS_FILE = join("embeddings", "gppl_embeddings.pkl")
MODEL_FILE = "models/gppl_model_{}.pkl"


def embed_sentences(sentences):
    model = SentenceTransformer('average_word_embeddings_glove.6B.300d', device="cuda")
    embeddings = np.asarray(model.encode(sentences))

    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump((sentences, embeddings), f)

    return embeddings


def get_scores_for_cat(start_cat=5, end_cat=-1):
    _, poem_scores = get_accuracy(start_cat, end_cat)
    return poem_scores


def save_scores(filename):
    global subset
    model_filename_ = format_model_filename(MODEL_FILE, "all", subset)
    scores = [get_accuracy(model_filename_)[1]]
    for idx in range(5, 15):
        model_filename_ = format_model_filename(MODEL_FILE, str(idx), subset)
        scores.append(get_accuracy(model_filename_, idx, idx + 1)[1])
    filepath = join("scores", filename)
    write_scores_to_file(scores, filepath)


def load_dataset(start_cat=5, end_cat=-1):
    global subset
    a1_sent_idx = []
    a2_sent_idx = []
    prefs_train = []
    sents = []

    if subset:
        with open(join(POEM_FOLDER, "multi_annotated_poems_10.txt"), "r") as f:
            multi_annotated = [line.strip() for line in f.readlines()]

    for filename in glob(join(POEM_FOLDER, "*.csv")):
        with open(filename, newline='') as f:
            lines = reader(f, dialect="unix")
            categories = next(lines)[start_cat:end_cat]
            for line in lines:
                if subset and line[1] not in multi_annotated and line[2] not in multi_annotated:
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
    a1_sent_idx = np.array(a1_sent_idx, dtype=int)
    a2_sent_idx = np.array(a2_sent_idx, dtype=int)
    prefs_train = np.array(prefs_train, dtype=float)

    return a1_sent_idx, a2_sent_idx, sent_features, prefs_train, ndims, sents


def train_model(start_cat=5, end_cat=-1, optimize=False):
    a1_train, a2_train, sent_features, prefs_train, ndims, _ = load_dataset(start_cat, end_cat)

    shape_s0 = len(a1_train) / 2.0
    rate_s0 = shape_s0 * np.var(prefs_train)
    model = GPPrefLearning(
        ndims, shape_s0=shape_s0, rate_s0=rate_s0)

    model.max_iter = 2000

    model.fit(a1_train, a2_train, sent_features, prefs_train,
              optimize=optimize, use_median_ls=True, input_type='zero-centered')
    return model


def get_accuracy(model_filename, start_cat=5, end_cat=-1):
    a1_train, a2_train, sent_features, prefs_train, ndims, sents = load_dataset(start_cat, end_cat)

    if not exists(model_filename):
        print("GPPL: no trained model found")
        return 0
    with open(model_filename, 'rb') as fh:
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


def print_all_accuracies():
    # Get Main accuracy accross all categories
    model_filename_ = format_model_filename(MODEL_FILE, "all", subset)
    print(f"GPPL acc all: {get_accuracy(model_filename_)[0]}")
    for i in range(5, 15):
        model_filename_ = format_model_filename(MODEL_FILE, str(i), subset)
        print(f"GPPL acc {i}: {get_accuracy(model_filename_, start_cat=i, end_cat=i + 1)[0]}")


def save_model(model_, filename):
    with open(filename, 'wb') as fh:
        pickle.dump(model_, fh)


if __name__ == "__main__":
    subset = True
    training = True

    subset_name = "_subset" if subset else ""

    if training:
        # Train on all categories
        model = train_model()
        model_filename = format_model_filename(MODEL_FILE, "all", subset)
        save_model(model, model_filename)
        # Train categories independently
        for cat in range(5, 15):
            model_filename = format_model_filename(MODEL_FILE, str(cat), subset)
            model = train_model(start_cat=cat, end_cat=cat + 1)
            save_model(model, model_filename)
        score_filename = f"gppl_scores{subset_name}.csv"
        save_scores(score_filename)
    print_all_accuracies()
