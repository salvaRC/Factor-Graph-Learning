import numpy as np
import pandas as pd
import pickle
import re, os, sys
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

AVAILABLEDATASETS = {'IMDB', 'Amazon', 'IMDB_unlabeled', 'twitter_disasters', 'YouTube',
                     'journalist_photographer', 'painter_architect', 'professor_physician', 'professor_teacher'}

LFTYPES = {'unigram', 'uni_and_bi_grams', 'MKNN', 'cluster'}
FEATURES = {'bow', 'tfidf', None}

FACTOR_MAP = {0: 'DEP_SIMILAR',
              1: 'DEP_FIXING',
              2: 'DEP_REINFORCING',
              3: 'DEP_EXCLUSIVE',
              4: 'DEP_NEGATED',
              5: 'DEP_FIXING2',
              6: 'DEP_NEGATED2',
              7: 'DEP_FIXING_PAPER',
              8: 'DEP_PRIORITY',
              9: 'DEP_REINFORCING2'}

REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

ENGLISH_SENTIMENT_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "co", "con",
    "could", "de", "describe", "detail", "did", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "few", "fifteen", "fifty", "fill",
    "find", "first", "for", "former", "formerly", "forty",
    "found", "from", "front", "full", "further", "get", "give", "go", "got",
    "had", "has", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely",
    "nevertheless", "next",
    "now", "of", "off", "often", "on",
    "once", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "several", "she", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "through", "throughout",
    "thru", "thus", "to", "together", "top", "toward", "towards",
    "twelve", "twenty", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"])


def print_progress(iteration, total, decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    frac = '%d/%d' % (iteration, total)
    sys.stdout.write('\r|%s|%s%s %s' % (bar, percents, '%', frac)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


class ProgressBar:
    def __init__(self, decimals=1, bar_length=100):
        self.decimals = str(decimals)
        self.bar_length = bar_length

    def update(self, iteration, total):
        str_format = "{0:." + self.decimals + "f}"
        percents = str_format.format(100.0 * (iteration / float(total)))
        filled_length = int(round(self.bar_length * iteration / float(total)))
        bar = '█' * filled_length + '-' * (self.bar_length - filled_length)
        frac = '%d/%d' % (iteration, total)
        return '|%s|%s%s %s' % (bar, percents, '%', frac)


def preprocess_reviews(reviews):
    """
    Function to clean review text
    """
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    return reviews


def load_dataset(dname='IMDB', lftype='unigram', testcorpus=False, processed_dir='processed', features='bow'):
    """
    Criterion of convergence
    
    Arguments:
        dname {string} -- Dataset name to load.
    
    Returns:
            -- Matrix of generated Labeling Functions (LFs)
            -- True train labels (to automate oracle)
            -- True test labels (to evaluate performance)
            -- Train features
            -- Test features
            -- train corpus in case humans are responding instead of automated oracle
    """
    if not dname in AVAILABLEDATASETS:
        print("%s not available" % dname)
        print("Choose from ", AVAILABLEDATASETS)

        raise ValueError("%s not available" % dname)

    if not lftype in LFTYPES:
        print("%s not available" % lftype)
        print("Choose from ", LFTYPES)
        raise ValueError("%s not available" % lftype)

    if not features in FEATURES:
        print("%s not available" % features)
        print("Choose from ", FEATURES)
        raise ValueError("%s not available" % features)

    corpus_test = []
    if lftype == 'cluster':
        LFs_generated, LFs_descriptions = pickle.load(
            open(os.path.join(processed_dir, '%s_clustering_LFs.pkl' % dname), 'rb'))
    elif lftype == 'unigram':
        LFs_generated, LFs_descriptions = pickle.load(
            open(os.path.join(processed_dir, '%s_unigram_LFs.pkl' % dname), 'rb'))
    elif lftype == 'uni_and_bi_grams':
        LFs_generated, LFs_descriptions = pickle.load(
            open(os.path.join(processed_dir, '%s_unibigram_LFs.pkl' % dname), 'rb'))
    elif lftype == 'MKNN':
        LFs_generated, LFs_descriptions = pickle.load(
            open(os.path.join(processed_dir, '%s_MKNN_LFs.pkl' % dname), 'rb'))
    if dname == 'IMDB':
        Ytrain = np.array([1 if i < 12500 else -1 for i in range(25000)])
        Ytest = Ytrain.copy()

        # Xtest = pickle.load(open('processed/imdb_bert_vecs_test.pkl','rb'))
        # Xtrain = pickle.load(open('processed/imdb_bert_vecs_train.pkl','rb'))
        try:
            with open('../data/imdb/aclImdb/movie_data/full_train.txt', 'r') as f:
                corpus = [line.strip() for line in f]
        except:
            with open(os.path.join(processed_dir, 'movie_data/full_train.txt'), 'r', encoding='utf-8') as f:
                corpus = [line.strip() for line in f]

        corpus = preprocess_reviews(corpus)
        if testcorpus:
            with open('../data/imdb/aclImdb/movie_data/full_test.txt', 'r') as f:
                corpus_test = [line.strip() for line in f]

            corpus_test = preprocess_reviews(corpus_test)
        labelmap = ['negative sentiment', 'positive sentiment']

    elif dname == 'IMDB_unlabeled':

        Ytrain = np.hstack(
            (np.array([1 if i < 12500 else -1 for i in range(25000)]), np.array([np.nan for i in range(50000)])))
        Ytest = np.array([1 if i < 12500 else -1 for i in range(25000)])

        # Xtest = pickle.load(open('processed/imdb_bert_vecs_test.pkl','rb'))
        # Xtrain = pickle.load(open('processed/imdb_bert_vecs_train.pkl','rb'))

        with open('../data/imdb/aclImdb/movie_data/full_train.txt', 'r') as f:
            reviews_train = [line.strip() for line in f]

        with open('../data/imdb/aclImdb/movie_data/full_unsup.txt', 'r') as f:
            reviews_unlbld = [line.strip() for line in f]

        corpus = preprocess_reviews(reviews_train) + preprocess_reviews(reviews_unlbld)

        if testcorpus:
            with open('../data/imdb/aclImdb/movie_data/full_test.txt', 'r') as f:
                corpus_test = [line.strip() for line in f]

            corpus_test = preprocess_reviews(corpus_test)
        labelmap = ['negative sentiment', 'positive sentiment']
    elif dname == 'twitter_disasters':

        df = pd.read_csv('../data/twitter_disaster/socialmedia-disaster-tweets-DFE_processed.csv')

        corpus = df[df.fold == 0].text.tolist()
        if testcorpus:
            corpus_test = df[df.fold == 1].text.tolist()

        Ytrain = df[df.fold == 0].choose_one.tolist()
        Ytest = df[df.fold == 1].choose_one.tolist()
        Ytrain = np.array([1 if x == 'Relevant' else -1 for x in Ytrain])
        Ytest = np.array([1 if x == 'Relevant' else -1 for x in Ytest])

        # Xtrain,Xtest = pickle.load(open('processed/twitter_bert_vecs.pkl','rb'))

        labelmap = ['Not relevant', 'Disaster related']
    elif dname == "Amazon":
        try:
            df = pd.read_csv('../data/AmazonReview/amazon_subset.csv')
        except:
            df = pd.read_csv(os.path.join(processed_dir, 'AmazonReview/amazon_subset.csv'))

        corpus = df[df.fold == 0].text.tolist()
        if testcorpus:
            corpus_test = df[df.fold == 1].text.tolist()

        Ytrain = df[df.fold == 0].label.to_numpy()
        Ytest = df[df.fold == 1].label.to_numpy()

        # Xtrain,Xtest = pickle.load(open('processed/amazon_bert_vecs.pkl','rb'))

        labelmap = ['negative sentiment', 'positive sentiment']
    elif dname == "YouTube":

        df = pd.read_csv('../data/YouTubeSpam/processed.csv')

        corpus = df[df.fold == 0].text.tolist()
        if testcorpus:
            corpus_test = df[df.fold == 1].text.tolist()

        Ytrain = df[df.fold == 0].label.to_numpy()
        Ytest = df[df.fold == 1].label.to_numpy()

        # Xtrain,Xtest = pickle.load(open('processed/YouTube_bert_vecs.pkl','rb'))

        labelmap = ['NOT spam', 'SPAM']
    elif dname == 'SuggestionMining':

        df = pd.read_csv('../data/SuggestionMining/SuggestionMining.csv')

        corpus = df[df.fold == 0].text.tolist()
        if testcorpus:
            corpus_test = df[df.fold == 1].text.tolist()

        Ytrain = df[df.fold == 0].label.to_numpy()
        Ytest = df[df.fold == 1].label.to_numpy()

        # Xtrain,Xtest = pickle.load(open('processed/SuggestionMining_bert_vecs.pkl','rb'))
        labelmap = ['Not a suggestion', 'Suggestion']
    elif dname in ['journalist_photographer', 'painter_architect', 'professor_physician', 'professor_teacher']:
        try:
            df = pd.read_csv(f'../data/BiasBios/{dname}.csv')
        except:
            df = pd.read_csv(os.path.join(processed_dir, f'BiasBios/{dname}.csv'))

        corpus = df[df.fold == 0].text.tolist()
        if testcorpus:
            corpus_test = df[df.fold == 1].text.tolist()

        Ytrain = df[df.fold == 0].label.to_numpy()
        Ytest = df[df.fold == 1].label.to_numpy()


        # Xtrain,Xtest = pickle.load(open('processed/SuggestionMining_bert_vecs.pkl','rb'))
        labelmap = ['Not a suggestion', 'Suggestion']

    Xtrain, Xtest = None, None
    if features == 'bow':
        Xtrain, Xtest = pickle.load(open(os.path.join(processed_dir, '%s_bow_svd_vecs.pkl' % dname), 'rb'))
        Xtrain = Xtrain.astype(np.float32)
        Xtest = Xtest.astype(np.float32)
    if features == 'tfidf':
        Xtrain, Xtest = pickle.load(open(os.path.join(processed_dir, '%s_tfidf_svd_vecs.pkl' % dname), 'rb'))
        Xtrain = Xtrain.astype(np.float32)
        Xtest = Xtest.astype(np.float32)

    return corpus, corpus_test, LFs_generated, LFs_descriptions, Ytrain, Ytest, Xtrain, Xtest, labelmap


def load_corpus(dname, test=False):
    if not dname in AVAILABLEDATASETS:
        print("%s not available" % dname)
        print("Choose from ", AVAILABLEDATASETS)
        raise ValueError

    if dname == 'IMDB':
        rtrain = None
        rtest = None
        with open('../data/imdb/aclImdb/movie_data/full_train.txt', 'r') as f:
            rtrain = [line.strip() for line in f]
        rtrain = preprocess_reviews(rtrain)
        if test:
            with open('../data/imdb/aclImdb/movie_data/full_test.txt', 'r') as f:
                rtest = [line.strip() for line in f]
            rtest = preprocess_reviews(rtest)

            return rtrain, rtest
        else:
            return rtrain, None

    elif dname == 'IMDB_unlabeled':
        rtrain = None
        rtest = None
        runlbld = None
        with open('../data/imdb/aclImdb/movie_data/full_train.txt', 'r') as f:
            rtrain = [line.strip() for line in f]
        rtrain = preprocess_reviews(rtrain)
        with open('../data/imdb/aclImdb/movie_data/full_unsup.txt', 'r') as f:
            runlbld = [line.strip() for line in f]
        runlbld = preprocess_reviews(runlbld)
        if test:
            with open('../data/imdb/aclImdb/movie_data/full_test.txt', 'r') as f:
                rtest = [line.strip() for line in f]
            rtest = preprocess_reviews(rtest)
            return rtrain + runlbld, rtest
        else:
            return rtrain + runlbld, None


    elif dname == 'twitter_disasters':
        df = pd.read_csv('../data/twitter_disaster/socialmedia-disaster-tweets-DFE_processed.csv')
        if test:
            return df[df.fold == 0].text.tolist(), df[df.fold == 1].text.tolist()
        else:
            return df[df.fold == 0].text.tolist(), None

    elif dname == "Amazon":
        amazondf = pd.read_csv('../data/AmazonReview/amazon_subset.csv')
        if test:
            return amazondf[amazondf.fold == 0].text.tolist(), amazondf[amazondf.fold == 1].text.tolist()
        else:
            return amazondf[amazondf.fold == 0].text.tolist(), None
    elif dname == 'YouTube':
        df = pd.read_csv('../data/YouTubeSpam/processed.csv')
        if test:
            return df[df.fold == 0].text.tolist(), df[df.fold == 1].text.tolist()
        else:
            return df[df.fold == 0].text.tolist(), None
    elif dname == 'SuggestionMining':
        df = pd.read_csv('../data/SuggestionMining/SuggestionMining.csv')
        if test:
            return df[df.fold == 0].text.tolist(), df[df.fold == 1].text.tolist()
        else:
            return df[df.fold == 0].text.tolist(), None


def evaluate_binary(X, Ytrue, verbose=False, is_ndarray=False, ignore_abstain=False, thresh=0, abstain=0):
    """
    Compute metrics for all labeling functions
    given the true binary labels.
    thresh: Decision boundary/threshold for deciding between pos (> thresh) or neg (< thresh) label
    """
    Ytrue = Ytrue.copy()
    if ignore_abstain:
        idxs = (X != abstain)
        X = X[idxs]
        Ytrue = Ytrue[idxs]
    if isinstance(Ytrue, list):
        Ytrue = np.array(Ytrue)
    if 0 in Ytrue and abstain != -1:
        Ytrue[Ytrue == 0] = -1

    isnan = np.isnan(Ytrue)
    if isnan.sum() > 0:
        if verbose:
            print('Handling unlabeled samples')
        # ignore unlabeled samples
        X = X.tocsr()[~isnan].tocoo()
        Ytrue = Ytrue[~isnan]

    numpos = np.count_nonzero(Ytrue > thresh)
    numneg = np.count_nonzero(Ytrue < thresh)
    n = X.shape[0]
    num_preds = 1 if len(X.shape) == 1 else X.shape[1]
    if is_ndarray:
        res = X.dot((np.matrix(Ytrue).T > thresh).astype(np.float32))
    else:
        res = X.multiply((np.matrix(Ytrue).T > thresh).astype(np.float32))
    tp = (res > thresh).sum(0)
    fn = (res < thresh).sum(0)

    if is_ndarray:
        res = X.dot(-(np.matrix(Ytrue).T < thresh).astype(np.float32))
    else:
        res = X.multiply(-(np.matrix(Ytrue).T < thresh).astype(np.float32))

    tn = (res > thresh).sum(0)
    fp = (res < thresh).sum(0)

    # fraction correct
    frac_correct = (tp + tn) / n
    frac_correct = np.asarray(frac_correct).flatten()

    # fraction incorrect
    frac_incorrect = (fp + fn) / n
    frac_incorrect = np.asarray(frac_incorrect).flatten()

    # (correct - incorrect) / n 
    frac_goodness = (tp + tn - fp - fn) / n
    frac_goodness = np.asarray(frac_goodness).flatten()

    # positives:recall 
    val = tp + fp
    val[val == 0] = 1.0
    precisionpos = tp / val
    recallpos = tp / numpos

    # negatives: specificity
    val = tn + fn
    val[val == 0] = 1.0
    precisionneg = tn / val
    recallneg = tn / numneg

    coverage = np.asarray((X != abstain).sum(0)).flatten() / X.shape[0]

    precisionpos = np.asarray(precisionpos).flatten()
    recallpos = np.asarray(recallpos).flatten()
    precisionneg = np.asarray(precisionneg).flatten()
    recallneg = np.asarray(recallneg).flatten()

    lfsign = np.asarray(np.sign((X - thresh).sum(0))).flatten()
    positive_funcs = lfsign > 0
    negative_funcs = lfsign < 0
    precision = np.zeros(num_preds)
    precision[negative_funcs] = precisionneg[negative_funcs]
    precision[positive_funcs] = precisionpos[positive_funcs]
    recall = np.zeros(num_preds)
    recall[negative_funcs] = recallneg[negative_funcs]
    recall[positive_funcs] = recallpos[positive_funcs]
    return coverage, precision, recall, frac_correct, frac_incorrect, frac_goodness


def eval_scores(y_true, y_scores, verbose=True):
    """
    Compute a number of metrics for predicted Y probabilities
    and return them. Optionally print them in verbose mode.
    """
    auc = roc_auc_score(y_true, y_scores)
    logloss = log_loss(y_true, y_scores)
    brier = np.square(y_true - y_scores).mean()
    if verbose:
        print('Brier score:', brier)
        print('Cross entropy loss:', logloss)
        print('AUC', auc)
    return auc, logloss, brier


def load_imdb_corpus(test=False, unlabeled=False):
    rtrain = None
    rtest = None
    runlbld = None
    with open('../data/imdb/aclImdb/movie_data/full_train.txt', 'r') as f:
        rtrain = [line.strip() for line in f]
    rtrain = preprocess_reviews(rtrain)
    if test:
        with open('../data/imdb/aclImdb/movie_data/full_test.txt', 'r') as f:
            rtest = [line.strip() for line in f]
        rtest = preprocess_reviews(rtest)
    if unlabeled:
        with open('../data/imdb/aclImdb/movie_data/full_unsup.txt', 'r') as f:
            runlbld = [line.strip() for line in f]
        runlbld = preprocess_reviews(runlbld)

    return rtrain, rtest, runlbld


def load_amazon_corpus(test=False):
    amazondf = pd.read_csv('../data/AmazonReview/amazon_subset.csv')
    if test:
        return amazondf.text.tolist()
    else:
        return amazondf[amazondf.fold == 0].text.tolist()


def load_twitter_corpus(test=False):
    df = pd.read_csv('../data/twitter_disaster/socialmedia-disaster-tweets-DFE_processed.csv')
    if test:
        return df.text.tolist()
    else:
        return df[df.fold == 0].text.tolist()


def loadinitialLFidxs(dname, lftype, dirname='processed'):
    initidxs = None
    with open(os.path.join(dirname, 'initLFs_%s_%s.txt' % (dname, lftype)), 'r') as f:
        initidxs = f.read()
        initidxs = list(map(int, initidxs.split(',')))
    return initidxs


def add_lfs(label_mat, lfs, Y=None, eval=True, return_metrics=False, cmp=None):
    if lfs is not None:
        label_mat_M = np.concatenate((label_mat, lfs), axis=1)
    else:
        label_mat_M = label_mat.copy()
    '''if eval:
        predsMV = get_majority_vote(label_mat_M, abstention_policy='abstain', abstention=0)
        print('MV preds') #, predsMV[mask])
        stats = eval_final_predictions(Y, predsMV, abstention=0, one_liner=True)
        label_matrixS, YtrainS = unmap_labels(label_mat_M, Y,
                                                        new_abstain=-1)  # needed for snorkel conventions
        label_model = LabelModel(cardinality=2, verbose=True)
        label_model.fit(label_matrixS, n_epochs=1000, log_freq=50, seed=77, lr=0.03)
        preds = label_model.predict(L=label_matrixS, tie_break_policy="abstain")
        probs = label_model.predict_proba(label_matrixS)
        print('DP preds:') #, preds_unmapped)
        eval_final_predictions(YtrainS, preds, probs=probs, abstention=-1, one_liner=True)
        if return_metrics:
            return label_mat_M, stats'''

    return label_mat_M


def change_labels(*args, new_label=-1, old_label=0):
    lst = []
    for arg in args:
        A = arg.copy()
        new_old = A == new_label
        A[A == old_label] = new_label
        A[new_old] = old_label
        if len(args) == 1:
            return A
        lst.append(A)

    return tuple(lst)


def append_stat_dfs(df1, df2):
    runs1 = df1['run'].nunique()
    df2['run'] = df2.run + runs1
    df = df1.append(df2, ignore_index=False)
    return df


def add_lf(L, lf):
    """

    :param L: Current label matrix
    :param lf: new LF outputs to be appended
    :return: L' = concat(L, lf)
    """
    return np.concatenate((L, lf), axis=1)


def to_int_label_array(X: np.ndarray, flatten_vector: bool = True) -> np.ndarray:
    """
    Taken from Snorkel v0.9
    Convert an array to a (possibly flattened) array of ints.

    Cast all values to ints and possibly flatten [n, 1] arrays to [n].
    This method is typically used to sanitize labels before use with analysis tools or
    metrics that expect 1D arrays as inputs.

    Parameters
    ----------
    X
        An array to possibly flatten and possibly cast to int
    flatten_vector
        If True, flatten array into a 1D array

    Returns
    -------
    np.ndarray
        The converted array

    Raises
    ------
    ValueError
        Provided input could not be converted to an np.ndarray
    """
    if np.any(np.not_equal(np.mod(X, 1), 0)):
        raise ValueError("Input contains at least one non-integer value.")
    X = X.astype(np.dtype(int))
    # Correct shape
    if flatten_vector:
        X = X.squeeze()
        if X.ndim == 0:
            X = np.expand_dims(X, 0)
        if X.ndim != 1:
            raise ValueError("Input could not be converted to 1d np.array")
    return X


def read_configs(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        configs = map(lambda line: tuple([float(i) for i in line.split()]), lines)

    return list(configs)

def selected_LFs(dataset="IMDB", direc="./"):
    if dataset == "IMDB":
        from factors_evaluation.imdb.my_lfs import selected_lfs
        return selected_lfs(direc=direc + "imdb/")
    elif dataset == "professor_teacher":
        from factors_evaluation.bb.my_lfs import selected_lfs
        return selected_lfs(direc=direc + "bb/")