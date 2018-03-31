""" Data loading and preparation """
import numpy as np


WORD_TYPES = ["VERB", "ADJ"]
LANGUAGES = ["es", "en"]

def mk_dictionary(lang):
    """ Load and preprocess word dictionary """
    print("loading", lang)
    full_dict = np.loadtxt(
        "../derived-dicts/%s-wik-20120320.dic" % lang,
        delimiter="\t",
        dtype=str
    )

    is_type = [full_dict[:, 1] == w_type for w_type in WORD_TYPES]

    reduced_dict = full_dict[np.any(is_type, axis=0)]

    if not np.sum(np.sum(t) for t in is_type) == reduced_dict.shape[0]:
        raise NotImplementedError("Handling of duplicate words not implemented")
    else:
        print("No duplicate words")

    np.random.shuffle(reduced_dict)

    return reduced_dict


def get_dictionaries():
    """ Get preprocessed dictionaries for all selected languages """
    return map(mk_dictionary, LANGUAGES)


def write_wordvec_dict(lang, outfile):
    count = 0
    vecs = dict()

    dt = np.dtype([
        ("word", str, 40),
        ("type", np.int32, 1),
        ("vec", np.float32, (300,))
    ])

    with open("../wiki.%s.vec" % lang, "r") as f:
        f.readline() # skip first line
        line = f.readline()
        while line:
            words = line[:-2].split(" ")
            # print(np.array(words[1:], dtype=float))
            vecs[words[0]] = np.array(words[1:], dtype=float)
            count += 1
            if count > 1000:
                break
            line = f.readline()
    return vecs

def get_sorted_wiki(lang):
    return np.load("../wiki_sorted.%s.npy" % lang)

def vecs_for_dict(word_dict, wiki_sorted):
    new_dt = np.dtype([
        ("word", str, 100),
        ("type", str, 10),
        ("vec", np.float32, (300,)),
    ])
    dict_words = word_dict[:, 0]
    wiki_words = wiki_sorted["word"]
    inds = np.searchsorted(wiki_words, dict_words, sorter=None)
    ind_is_good = dict_words == wiki_words[inds]
    dict_words_accepted = word_dict[ind_is_good, :]
    good_inds = inds[ind_is_good]

    count = good_inds.shape[0]
    return np.array(
        [(
            dict_words_accepted[ind, 0],
            dict_words_accepted[ind, 1],
            wiki_sorted["vec"][good_inds[ind]]
        ) for ind in range(count)],
        dtype=new_dt,
    )


SPANISH_DICT = None
ENGLISH_DICT = None

if __name__ == "__main__":
    # en_wordvec_dict = mk_wordvec_dict("en")
    # SPANISH_DICT, ENGLISH_DICT = get_dictionaries()
    ENGLISH_DICT = mk_dictionary("es")
    en_wiki = get_sorted_wiki("es")
    final_vecs = vecs_for_dict(ENGLISH_DICT, en_wiki)
    # print(SPANISH_DICT, ENGLISH_DICT)
