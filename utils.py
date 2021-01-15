POEM_FOLDER = "data_poems/"


def arrange_poem_scores(scores):
    scores_per_poem = {}
    for score_list in scores:
        for poem in score_list:
            curr_list = scores_per_poem.get(poem, [])
            curr_list.append(score_list[poem])
            scores_per_poem[poem] = curr_list
    return scores_per_poem