from os.path import join

from bertranker_datasets import PairPrefDataset, SingleDataset
from torch.utils.data import DataLoader
from csv import reader
import torch
import csv
from utils import POEM_FOLDER, write_scores_to_file, format_model_filename, parse_arguments_for_methods
from code_bertranker.BERT_cQA import train_bertcqa, predict_bertcqa, BertRanker

MODEL_FILE = "models/bertranker_{}.pt"

def train_model(start_cat=5, end_cat=-1):
    dataset, _ = get_dataset(start_cat, end_cat)
    dataloader = DataLoader(dataset, batch_size=24)
    return train_bertcqa(dataloader, nepochs=10)


def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)


def load_model(path):
    model = BertRanker()
    model.load_state_dict(torch.load(path))
    model.to("cuda")
    return model


def get_scores_for_cat(start_cat=5, end_cat=-1):
    _, poem_scores = get_accuracy(start_cat, end_cat)
    return poem_scores


def save_scores(filename):
    global subset
    model_filename = format_model_filename(MODEL_FILE, "all", subset)
    # Scores from all 
    scores = [get_accuracy(model_filename)[1]]
    for i in range(5, 15):
        model_filename = format_model_filename(MODEL_FILE, str(i), subset)
        scores.append(get_accuracy(model_filename, i, i + 1)[1])

    filepath = join("scores", filename)
    write_scores_to_file(scores, filepath)


def get_dataset(start_cat=5, end_cat=-1):
    global subset
    if subset:
        with open(join(POEM_FOLDER, "multi_annotated_poems_10.txt"), "r") as f:
            multi_annotated = [line.strip() for line in f.readlines()]

    poem_pairs = []
    targets = []
    poems = []
    with open(join(POEM_FOLDER, "consolidated_batches.csv"), "r") as f:
        csv_reader = reader(f)
        # First line contains csv column names
        header = next(csv_reader)
        categories = header[start_cat:end_cat]
        for line in csv_reader:
            if subset and line[1] not in multi_annotated and line[2] not in multi_annotated:
                continue
            poem1 = line[1].replace("<br>", "\n")
            poem2 = line[2].replace("<br>", "\n")
            for idx, cat_ in enumerate(categories, start_cat):
                label = line[idx]
                if label.strip() == "1":
                    target = 1
                elif label.strip() == "2":
                    target = -1
                elif label.strip():
                    target = 0
                else:
                    continue
                poem_pairs.append([poem1, poem2])
                poems.extend([poem1, poem2])
                targets.append(target)
    # Make each poem appear only once
    poems = list(set(poems))
    return PairPrefDataset(poem_pairs, targets), poems, poem_pairs, targets


def get_accuracy(model_filename, start_cat=5, end_cat=-1):
    _, poems, poem_pairs, targets = get_dataset(start_cat, end_cat)
    single_ds = SingleDataset(poems)
    dataloader = DataLoader(single_ds, batch_size=24)
    scores, embs = predict_bertcqa(load_model(model_filename), dataloader, "cuda")
    correct = 0.
    wrong = 0.
    poem_scores = {}
    for i in range(len(poem_pairs)):
        poem1 = poem_pairs[i][0]
        poem2 = poem_pairs[i][1]
        poem1_idx = single_ds.get_index_of_poem(poem1)
        poem2_idx = single_ds.get_index_of_poem(poem2)
        poem1_score = scores[poem1_idx]
        poem2_score = scores[poem2_idx]

        if poem1_score > poem2_score and targets[i] == 1:
            correct += 1
        elif poem2_score > poem1_score and targets[i] == -1:
            correct += 1
        elif poem1_score == poem2_score:
            continue
        else:
            wrong += 1

        if poem1 not in poem_scores:
            poem_scores[poem1] = poem1_score
        if poem2 not in poem_scores:
            poem_scores[poem2] = poem2_score
    return correct / (correct + wrong), poem_scores


def print_all_accuracies():
    global subset
    # Get Main accuracy accross all categories
    model_filename = format_model_filename(MODEL_FILE, "all", subset)
    print(f"BertGPPL acc all: {get_accuracy(model_filename)[0]}")
    for i in range(5, 15):
        model_filename = format_model_filename(MODEL_FILE, str(i), subset)
        print(f"BertGPPL acc {i}: {get_accuracy(model_filename, start_cat=i, end_cat=i + 1)[0]}")


if __name__ == '__main__':
    args = parse_arguments_for_methods()
    subset = args.subset
    training = not args.no_traing

    subset_name = "_subset" if subset else ""

    if training:
        # Train on all categories
        model_filename = format_model_filename(MODEL_FILE, "all", subset)
        model, device = train_model()
        save_model(model, filepath=model_filename)
        # iterate over all categories independently
        for cat in range(5, 15):
            model_filename = format_model_filename(MODEL_FILE, "all", subset)
            model, device = train_model(start_cat=cat, end_cat=cat + 1)
            save_model(model, model_filename)
            # Remove model to allow multiple trainings after each other
            del model
        save_scores(f"bertranker_scores{subset_name}.csv")
    print_all_accuracies()
