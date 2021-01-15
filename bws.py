from glob import glob
from os import getcwd
from os.path import join
from collections import Counter, OrderedDict
from csv import reader
import csv
from utils import arrange_poem_scores, POEM_FOLDER

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
    with open("scores/bws_subset.csv", "w+") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
        for poem in scores_per_poem:
            line = [poem.replace("\n", "<br>")] + scores_per_poem[poem]
            csv_writer.writerow(line)

def load_dataset(start_cat=5, end_cat=-1, subset=True):
    if subset:
        with open(join(POEM_FOLDER, "multi_annotated_poems_10.txt"), "r") as f:
            multi_annotated = [line.strip() for line in f.readlines()]

    most_positive = OrderedDict()
    most_negative = OrderedDict()
    total = OrderedDict()
    sents = list()
    pairs = list()

    for filename in glob(join(POEM_FOLDER, "*.csv")):
        with open(filename, newline='') as f:
            lines = reader(f, dialect="unix")
            next(lines)
            for line in lines:
                # Skip line if training should be done only on a subset and line does not contain any related poem
                if subset and line[1] not in multi_annotated and line[2] not in multi_annotated:
                    continue
                # Replace <br> token with unix line ending
                line = [col.replace("<br>", "\n").strip() for col in line]
                left_poem, right_poem = line[1:3]
                for text in line[1:3]:
                    if text not in sents:
                        sents.append(text)
                labels = line[start_cat:end_cat]
                for poem in [left_poem, right_poem]:
                    for counter in [most_positive, most_negative, total]:
                        if poem not in counter:
                            counter[poem] = 0
                for label in labels:
                    if label.strip() == "1":
                        most_positive[left_poem] += 1
                        most_negative[right_poem] += 1
                        pairs.append((left_poem, right_poem, 1))
                    elif label.strip() == "2":
                        most_positive[right_poem] += 1
                        most_negative[left_poem] += 1
                        pairs.append((left_poem, right_poem, -1))
                    if label.strip():
                        total[right_poem] += 1
                        total[left_poem] += 1
    return most_positive, most_negative, total, sents, pairs

def compute_scores():
    return compute_scores_with_pairs()[0]

def compute_scores_with_pairs(start_cat=5, end_cat=-1):
    most_positive, most_negative, total, _, pairs = load_dataset(start_cat, end_cat)
    scores = list()
    sents = []
    for poem in most_positive.keys():
        scores.append((most_positive[poem] - most_negative[poem]) /
                      max((total[poem], 1)))
        sents.append(poem)
    return scores, sents, pairs

def get_accuracy(start_cat=5, end_cat=-1):
    scores, sents, pairs = compute_scores_with_pairs(start_cat, end_cat)
    correct = 0.
    wrong = 0.
    poem_scores = {}
    for pair in pairs:
        poem1, poem2, pref = pair
        poem1_idx = sents.index(poem1)
        poem2_idx = sents.index(poem2)
        poem1_score = scores[poem1_idx]
        poem2_score = scores[poem2_idx]
        if poem1_score > poem2_score and pref == 1:
            correct += 1
        elif poem2_score > poem1_score and pref == -1:
            correct += 1
        elif poem1_score == poem2_score:
            wrong += 1
        else:
            wrong += 1

        if poem1 not in poem_scores:
            poem_scores[poem1] = poem1_score
        if poem2 not in poem_scores:
            poem_scores[poem2] = poem2_score
    return correct/(correct+wrong), poem_scores

def print_all_accuracies():
    print(f"BWS acc all: {get_accuracy()[0]}")
    for i in range(5, 15):
        print(f"BWS acc {i}: {get_accuracy(i, i+1)[0]}")


if __name__ == "__main__":
    print(compute_scores())
    print_all_accuracies()
    save_scores()
    load_dataset(5,6)
