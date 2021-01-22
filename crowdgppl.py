import pickle
from sys import path
from glob import glob
from os import getcwd
from os.path import join, realpath, dirname, exists
from sentence_transformers import SentenceTransformer
from csv import reader
import logging
import numpy as np

__location__ = realpath(join(getcwd(), dirname(__file__)))
path.append(join(__location__, "code_gppl", "python", "models"))

from code_gppl.python.models.collab_pref_learning_svi import CollabPrefLearningSVI
from utils import POEM_FOLDER, write_scores_to_file, format_model_filename, parse_arguments_for_methods

MODEL_FILE = "models/crowdgppl_model_{}.pkl"


def embed_sentences(sentences):
    sentence_encoder = SentenceTransformer('average_word_embeddings_glove.6B.300d', device="cuda")
    embeddings = np.asarray(sentence_encoder.encode(sentences))
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


def save_model(model_, filename):
    with open(filename, 'wb') as fh:
        pickle.dump(model_, fh)


def load_dataset(start_cat=5, end_cat=-1):
    global subset
    person_train = []
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
            # categories = next(lines)[3:]
            for line in lines:
                if subset and line[1] not in multi_annotated and line[2] not in multi_annotated:
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
    return person_train_idx, a1_sent_idx, a2_sent_idx, sent_features, prefs_train, ndims, sents


def train_model(start_cat=5, end_cat=-1):
    person_train_idx, a1_train, a2_train, sent_features, prefs_train, ndims, _ = load_dataset(start_cat, end_cat)

    model_ = CollabPrefLearningSVI(ndims, shape_s0=2, rate_s0=200, use_lb=True,
                                   use_common_mean_t=True, ls=None)

    model_.max_iter = 2000

    model_.fit(person_train_idx, a1_train, a2_train, sent_features, prefs_train, optimize=False,
               use_median_ls=True, input_type='zero-centered')

    logging.info("**** Completed training CrowdGPPL ****")
    return model_


def get_accuracy(model_path, start_cat=5, end_cat=-1):
    person_train_idx, a1_train, a2_train, sent_features, prefs_train, ndims, sents = load_dataset(start_cat, end_cat)

    if not exists(model_path):
        print("CrowdGPPL: no trained model found")
        return 0
    # Get Scores
    with open(model_path, 'rb') as fh:
        model_ = pickle.load(fh)
        scores = model_.predict_f()
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
    global subset
    # Get accuracy accross all categories
    model_filename_ = format_model_filename(MODEL_FILE, "all", subset)
    all_acc, _ = get_accuracy(model_filename_)
    print(f"CrowdGPPL acc all: {all_acc}")
    for cat_idx in range(5, 15):
        model_filename_ = format_model_filename(MODEL_FILE, str(cat_idx), subset)
        cat_acc, _ = get_accuracy(model_filename_, start_cat=cat_idx, end_cat=cat_idx + 1)
        print(f"CrowdGPPL acc {cat_idx}: {cat_acc}")


if __name__ == "__main__":
    args = parse_arguments_for_methods()
    subset = args.subset
    training = not args.no_training

    subset_name = "_subset" if subset else ""

    if training:
        # Train all categories together
        model_filename = format_model_filename(MODEL_FILE, "all", subset)
        model = train_model()
        save_model(model, model_filename)
        # Train categories independently
        for i in range(5, 15):
            model_filename = format_model_filename(MODEL_FILE, str(i), subset)
            model = train_model(start_cat=i, end_cat=i + 1)
            save_model(model, model_filename)
        save_scores(f"crowdgppl{subset_name}.csv")
    print_all_accuracies()
