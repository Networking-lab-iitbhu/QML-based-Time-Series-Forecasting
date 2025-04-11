from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api

model = api.load("glove-twitter-25")
import nltk
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
import numpy as np
from tqdm import tqdm
import random

random.seed(0)
# Uncomment on First Run
try:
    nltk.data.find("tokenizers/stopwords")
except LookupError:
    nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


def scale_data(data, scale=None, dtype=np.float32):
    if scale is None:
        scale = [0, 1]
    min_data, max_data = [float(np.min(data)), float(np.max(data))]
    min_scale, max_scale = [float(scale[0]), float(scale[1])]
    all_data = []
    for d in data:
        scaled_data = (
            (max_scale - min_scale) * (d - min_data) / (max_data - min_data)
        ) + min_scale
        scaled_data = scaled_data.astype(dtype)
        all_data.append(scaled_data)
    return all_data


def scale_2d(data, min_data, max_data, dtype=np.float32):
    scale = [0, 1]
    all_data = []
    for d in tqdm(data):
        this_data = []
        for j in d:
            scaled_data = (
                (scale[1] - scale[0]) * (j - min_data) / (max_data - min_data)
            ) + scale[0]
            scaled_data = scaled_data.astype(dtype)
            this_data.append(scaled_data)
        all_data.append(this_data)
    return all_data


def get_pca_imdb(pca_dims, text):
    flattened_words_corpus = []
    for document in text:
        for word in document.split(" "):
            if word.lower() not in stop_words and word.lower() in model:
                flattened_words_corpus.append(model[word.lower()])
    pca_model = PCA(pca_dims)
    pca_model.fit(flattened_words_corpus)
    return pca_model


def glove_from_dataset(data, labels, args, pca=False):
    glove_corpus = []
    flattened = []
    bad_indicies = []
    for count, document in tqdm(enumerate(data)):
        glove_document = []
        for word in document.split(" "):
            if word.lower() not in stop_words and word.lower() in model:
                if pca is False:
                    glove_word = model[word.lower()][:16]
                else:
                    glove_word = pca.transform(model[word.lower()].reshape([1, -1]))[0]
                glove_document.append(glove_word)
        if len(glove_document) < 3:
            bad_indicies.append(count)
        else:
            if args.pad_mode =='on' or args.pad_mode =='selective':
                while len(glove_document) < args.sentence_len:
                    glove_document.append(np.zeros_like(glove_word))
            glove_corpus.append(glove_document)
            for i in range(len(glove_document)):
                flattened.append(glove_document[i])

    # Normalize
    min_data, max_data = [float(np.min(flattened)), float(np.max(flattened))]
    flattened = scale_data(flattened)
    glove_corpus = scale_2d(glove_corpus, min_data, max_data)
    return (glove_corpus, flattened), [
        labels[i] for i in range(len(labels)) if i not in bad_indicies
    ]


def get_lam_text(args, pca, text):
    flattened_words_corpus = []
    unflattened_words = []
    for document in text:
        doc_words = []
        for word in document.split(" "):
            if (
                word.lower() not in stop_words
                and word.lower() in model # glove-twitter-25
                and word.lower()
            ):
                flattened_words_corpus.append(model[word.lower()][:pca])
                doc_words.append(model[word.lower()][:pca])
        if args.pad_mode =='on' or args.pad_mode =='selective':
            while (len(doc_words)) < 10:
                doc_words.append(np.zeros_like(model[word.lower()][:pca]))
        unflattened_words.append(doc_words)
    return (unflattened_words, flattened_words_corpus)


# def get_lambeq(args, pca=False, split="train", ds="mc"):
#     # From https://cqcl.github.io/lambeq/tutorials/trainer_classical.html#Input-data
#     def read_data(filename):
#         labels, sentences = [], []
#         with open(filename) as f:
#             for line in f:
#                 t = float(line[0])
#                 labels.append([t, 1 - t])
#                 sentences.append(line[1:].strip())
#         return labels, sentences

#     if ds == "mc":
#         train_labels, train_data = read_data("datasets/original/mc_train_data.txt")
#         val_labels, val_data = read_data("datasets/original/mc_dev_data.txt")
#         test_labels, test_data = read_data("datasets/original/mc_test_data.txt")
#     else:
#         train_labels, train_data = read_data("datasets/original/rp_train_data.txt")
#         test_labels, test_data = read_data("datasets/original/rp_test_data.txt")
#     if split == "train":
#         return get_lam_text(args, pca, train_data), [int(i[0]) for i in train_labels]
#     if split == "test" or split == "val":
#         return get_lam_text(args, pca, test_data), [int(i[0]) for i in test_labels]


# def get_uci(args, pca=False, split="train", ds="amazon"):
#     def read_uci(filename):
#         labels, sentences = [], []
#         with open(filename) as f:
#             lines = [line for line in f]
#             random.shuffle(lines)
#             for line in lines:
#                 t = int(line[-2])
#                 labels.append(t)
#                 sentences.append(line[:-3].strip())
#         return labels, sentences

#     if ds == "amazon":
#         labels, data = read_uci("datasets/original/amazon_cells_labelled.txt")
#     elif ds == "imdb_labelled":
#         labels, data = read_uci("datasets/original/imdb_labelled.txt")
#     elif ds == "yelp":
#         labels, data = read_uci("datasets/original/yelp_labelled.txt")
#     train_data = data[: int(len(data) * 0.7)]
#     train_labels = labels[: int(len(data) * 0.7)]
#     test_data = data[int(len(data) * 0.7) : int(len(data) * 0.9)]
#     test_labels = labels[int(len(data) * 0.7) : int(len(data) * 0.9)]
#     val_data = data[int(len(data) * 0.9) :]
#     val_labels = labels[int(len(data) * 0.9) :]
#     pca_model = get_pca_imdb(pca, data)
#     if split == "train":
#         return glove_from_dataset(train_data, train_labels, args, pca_model)
#     if split == "test":
#         return glove_from_dataset(test_data, test_labels, args, pca_model)
#     if split == "val":
#         return glove_from_dataset(val_data, val_labels, args, pca_model)


# def get_dataset(args, split):
#     import os

#     print(f"Processing Dataset: {args.dataset} {split}")
#     if args.dataset == "lambeq1":
#         out = get_lambeq(args, args.pca_dims, split=split, ds="mc")
#         return out
#     elif args.dataset == "lambeq2":
#         out = get_lambeq(args, args.pca_dims, split=split, ds="rp")
#         return out
#     elif args.dataset in ["amazon", "imdb_labelled", "yelp"]:
#         if os.path.isfile(f"datasets/cached/{args.dataset}_{split}_{args.pad_mode}.npy"):
#             return np.load(
#                 f"datasets/cached/{args.dataset}_{split}_{args.pad_mode}.npy", allow_pickle=True
#             )
#         else:
#             out = get_uci(args, args.pca_dims, split=split, ds=args.dataset)
#             out = np.array(out, dtype=object)
#             np.save(f"datasets/cached/{args.dataset}_{split}_{args.pad_mode}.npy", out)
#             return out
import os
import numpy as np
import random

def get_lambeq(args, pca=False, split="train", ds="mc"):
    # From https://cqcl.github.io/lambeq/tutorials/trainer_classical.html#Input-data
    def read_data(filename):
        labels, sentences = [], []
        with open(filename) as f:
            for line in f:
                t = float(line[0])
                labels.append([t, 1 - t])
                sentences.append(line[1:].strip())
        return labels, sentences

    base_path = os.path.expanduser("~/quantum-ml-main/quantum-ml-main/datasets/original")
    
    if ds == "mc":
        train_labels, train_data = read_data(os.path.join(base_path, "mc_train_data.txt"))
        val_labels, val_data = read_data(os.path.join(base_path, "mc_dev_data.txt"))
        test_labels, test_data = read_data(os.path.join(base_path, "mc_test_data.txt"))
    else:
        train_labels, train_data = read_data(os.path.join(base_path, "rp_train_data.txt"))
        test_labels, test_data = read_data(os.path.join(base_path, "rp_test_data.txt"))
    
    if split == "train":
        return get_lam_text(args, pca, train_data), [int(i[0]) for i in train_labels]
    if split == "test" or split == "val":
        return get_lam_text(args, pca, test_data), [int(i[0]) for i in test_labels]


def get_uci(args, pca=False, split="train", ds="amazon"):
    def read_uci(filename):
        labels, sentences = [], []
        with open(filename) as f:
            lines = [line for line in f]
            random.shuffle(lines)
            for line in lines:
                t = int(line[-2])
                labels.append(t)
                sentences.append(line[:-3].strip())
        return labels, sentences

    base_path = os.path.expanduser("~/quantum-ml-main/quantum-ml-main/datasets/original")
    
    if ds == "amazon":
        labels, data = read_uci(os.path.join(base_path, "amazon_cells_labelled.txt"))
    elif ds == "imdb_labelled":
        labels, data = read_uci(os.path.join(base_path, "imdb_labelled.txt"))
    elif ds == "yelp":
        labels, data = read_uci(os.path.join(base_path, "yelp_labelled.txt"))

    train_data = data[: int(len(data) * 0.7)]
    train_labels = labels[: int(len(data) * 0.7)]
    test_data = data[int(len(data) * 0.7) : int(len(data) * 0.9)]
    test_labels = labels[int(len(data) * 0.7) : int(len(data) * 0.9)]
    val_data = data[int(len(data) * 0.9) :]
    val_labels = labels[int(len(data) * 0.9) :]
    
    pca_model = get_pca_imdb(pca, data)
    
    if split == "train":
        return glove_from_dataset(train_data, train_labels, args, pca_model)
    if split == "test":
        return glove_from_dataset(test_data, test_labels, args, pca_model)
    if split == "val":
        return glove_from_dataset(val_data, val_labels, args, pca_model)


def get_dataset(args, split):
    print(f"Processing Dataset: {args.dataset} {split}")

    cache_path = os.path.expanduser("~/quantum-ml-main/quantum-ml-main/datasets/cached")
    os.makedirs(cache_path, exist_ok=True)  # ✅ Ensure cache directory exists

    cache_file = os.path.join(cache_path, f"{args.dataset}_{split}_{args.pad_mode}.npy")

    if args.dataset == "lambeq1":
        return get_lambeq(args, args.pca_dims, split=split, ds="mc")
    elif args.dataset == "lambeq2":
        return get_lambeq(args, args.pca_dims, split=split, ds="rp")
    elif args.dataset in ["amazon", "imdb_labelled", "yelp"]:
        if os.path.isfile(cache_file):
            return np.load(cache_file, allow_pickle=True)
        else:
            out = get_uci(args, args.pca_dims, split=split, ds=args.dataset)
            out = np.array(out, dtype=object)
            np.save(cache_file, out)
            return out
