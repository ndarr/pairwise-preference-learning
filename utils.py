from csv import reader, writer
from os.path import join
from argparse import ArgumentParser

POEM_FOLDER = "data_poems/"


def arrange_poem_scores(scores):
    """
    Merges a list of dictionaries containing poems and their respective score into one large dictionary
    :param    list of dictionaries containing poems and scores
    :returns  dictionary with all dictionaries put togeteher 
    """
    scores_per_poem = {}
    for score_list in scores:
        for poem in score_list:
            curr_list = scores_per_poem.get(poem, [])
            curr_list.append(score_list[poem])
            scores_per_poem[poem] = curr_list
    return scores_per_poem


def get_poem_pairs_with_labels(filename):
    """
    :param     filename   name of csv in POEM_FOLDER
    :returns   
        pairs: list containing all pairs in the dataset with their annotations.
               each element is a dictionary with keys: poem1, poem2, dataset1, dataset2 and the various categories + all
        poems: list of all used poems with duplicates if they were used multiple times
    """
    # Load pairs
    poems = []
    pairs = []
    with open(join(POEM_FOLDER, filename), "r") as f:
        csv_reader = reader(f)
        header = next(csv_reader)
        categories = header[5:]
        for line in csv_reader:
            poem1 = line[1]
            poem2 = line[2]

            dataset1 = line[3]
            dataset2 = line[4]

            pair = {}
            pair["poem1"] = poem1
            pair["poem2"] = poem2
            pair["dataset1"] = dataset1
            pair["dataset2"] = dataset2
            poems.extend([poem1, poem2])
            all_ = []
            for idx, cat in enumerate(categories, 5):
                label = line[idx]
                if label.strip() == "1":
                    target = 1
                elif label.strip() == "2":
                    target = -1
                elif label.strip():
                    target = 0
                else:
                    continue
                pair[cat] = target
                all_.append(target)
            pair['all'] = all_
            pairs.append(pair)
    return pairs, poems


def write_scores_to_file(scores, filepath):
    scores_per_poem = arrange_poem_scores(scores)
    header = ['poem', 'all', 'coherent', 'grammatical', 'melodious', 'moved', 'real', 'rhyming',
              'readable', 'comprehensible', 'intense', 'liking']
    with open(filepath, "w+") as f:
        csv_writer = writer(f)
        csv_writer.writerow(header)
        for poem in scores_per_poem:
            line = [poem.replace("\n", "<br>")] + scores_per_poem[poem]
            csv_writer.writerow(line)


def parse_arguments_for_methods():
    """
    Parses System Arguments and returns them as a dictionary
    :return: dict with boolean entries for no-training and subset
    """
    argparser = ArgumentParser()
    argparser.add_argument("--no-training", dest="training", default=False, action="store_true")
    argparser.add_argument("--subset", dest="subset", default=False, action="store_true")
    args = argparser.parse_args()
    return args


def format_model_filename(basename, category, subset):
    """
    :param basename     String to format with one format place
    :param category     category identifier as string
    :param subset       Flag if subset
    :return:            formatted filename of model
    """
    subset_name = "_subset" if subset else ""
    # Train all categories together
    model_filename = basename.format(category + subset_name)
    return model_filename
