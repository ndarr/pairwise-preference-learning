import gppl
import crowdgppl
import bws
import bertranker
from scipy.stats import spearmanr

def compute_correlation():
    bws_scores = "BWS", bws.compute_scores()
    crowdgppl_scores = "crowdGPPL", crowdgppl.compute_scores()
    gppl_scores = "GPPL", gppl.compute_scores()
    bertranker_scores = "BERTRanker", bertranker.compute_scores()

    print(len(bws_scores[1]))
    print(len(gppl_scores[1]))
    print(len(crowdgppl_scores[1]))
    print(len(bertranker_scores[1]))

    results = sorted([bws_scores, gppl_scores, crowdgppl_scores, bertranker_scores])
    tuples = [sorted((pair1, pair2)) for pair1 in results for pair2 in results
              if pair1 != pair2 and pair1[0] < pair2[0]]
    for (type1, scores1), (type2, scores2) in tuples:
        correlation = spearmanr(scores1, scores2)
        print(f"Correlation between {type1} and {type2}: {correlation[0]}")

if __name__ == "__main__":
    # get_best_poems()
    # compute_correlation()
    # plot_gppl()
    # plot_lengthscales()
    # gppl.get_all_accuracies()
    # crowdgppl.get_all_accuracies()
    # bws.get_all_accuracies()
    # bertgppl.get_all_accuracies()
    bws.save_scores()