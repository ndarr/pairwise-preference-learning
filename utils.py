from csv import reader
from os.path import join

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
    