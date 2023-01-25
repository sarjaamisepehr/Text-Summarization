"""
Created on Jan 6 2023
@author: Sepehr Sarjami
"""


# First Hand Imports: Prerequisits
import math
from typing import List, Dict

def add_alpha(sentences, n=3):
    def transform(score):
        return math.exp(score * 5)

    n = min(n, max(1, int(len(sentences) * 0.5)))
    scores = [transform(x.score) for x in sentences]
    # Note: this does not consider collision
    thres = sorted(scores)[-n]

    min_score = min(scores)
    max_score = max(scores)
    span = max_score - min_score + 1
    for sent in sentences:
        sent.transformed_score = round(
            (transform(sent.score) - min_score + 1) / span, 4) * 50
        sent.alpha = sent.transformed_score / 50
        if transform(sent.score) < thres:
            sent.alpha = 0

#pip install spacy
import spacy
spacy.cli.download("en_core_web_sm")

from summa.preprocessing.textcleaner import init_textcleanner, filter_words, merge_syntactic_units

NLP = spacy.load("en_core_web_sm")


def split_sentences(text):
    doc = NLP(text)
    return [sent.text.strip() for sent in doc.sents]


def clean_text_by_sentences(text, additional_stopwords=None):
    """Tokenizes text into sentences, 
    applying filters and lemmatizing them.
    Returns a SyntacticUnit list."""
    init_textcleanner("english", additional_stopwords)
    original_sentences = split_sentences(text)
    filtered_sentences = filter_words(original_sentences)
    return merge_syntactic_units(original_sentences, filtered_sentences)


# Second Hand Imports: Keywords
from typing import List, Dict, Tuple, Optional
import summa.graph
import math
from typing import List, Dict

from langdetect import detect
from summa.pagerank_weighted import pagerank_weighted_scipy as _pagerank
from summa.preprocessing.textcleaner import clean_text_by_word as _clean_text_by_word
from summa.preprocessing.textcleaner import tokenize_by_word as _tokenize_by_word
from summa.commons import build_graph as _build_graph
from summa.commons import remove_unreachable_nodes as _remove_unreachable_nodes
from summa.keywords import (
    _get_words_for_graph, _set_graph_edges, _lemmas_to_words
)
import summa.graph

def _extract_tokens(lemmas, scores) -> List[Tuple[float, str]]:
    lemmas.sort(key=lambda s: scores[s], reverse=True)
    return [(scores[lemmas[i]], lemmas[i]) for i in range(len(lemmas))]

def keywordsFunc(
        text: str, deaccent: bool = False,
        additional_stopwords: List[str] = None) -> Tuple[
            List[Tuple[float, str]], Optional[Dict[str, List[str]]],
            Optional[summa.graph.Graph], Dict[str, float]]:
    if not isinstance(text, str):
        raise ValueError("Text parameter must be a Unicode object (str)!")

    # Gets a dict of word -> lemma
    lang = "en"
    tokens = _clean_text_by_word(
        text, "english", deacc=deaccent,
        additional_stopwords=additional_stopwords)
    split_text = list(_tokenize_by_word(text))
    # Creates the graph and adds the edges
    graph = _build_graph(_get_words_for_graph(tokens))
    _set_graph_edges(graph, tokens, split_text)
    del split_text  # It's no longer used

    _remove_unreachable_nodes(graph)

    # PageRank cannot be run in an empty graph.
    if not graph.nodes():
        return [], {}, None, {}

    # Ranks the tokens using the PageRank algorithm. Returns dict of lemma -> score
    pagerank_scores = _pagerank(graph)

    extracted_lemmas = _extract_tokens(graph.nodes(), pagerank_scores)

    lemmas_to_word = None
    lemmas_to_word = _lemmas_to_words(tokens)

    return extracted_lemmas, lemmas_to_word, graph, pagerank_scores

# Third Hand Imports: Sentences & Summarization

"""Using similarity function from the original TextRank algorithm."""
#pip install langdetect
from langdetect import detect
from summa.pagerank_weighted import pagerank_weighted_scipy as _pagerank
from summa.preprocessing.textcleaner import clean_text_by_sentences as _clean_text_by_sentences

from summa.commons import remove_unreachable_nodes as _remove_unreachable_nodes
from summa.summarizer import _set_graph_edge_weights, _add_scores_to_sentences

def summarize(text, additional_stopwords=None):
    if not isinstance(text, str):
        raise ValueError("Text parameter must be a Unicode object (str)!")

    lang = "en"
    paragraphs = text.split("\n")
    sentences = []
    paragraph_index = 0
    for paragraph in paragraphs:
        # Gets a list of processed sentences.
        if paragraph:
            tmp = clean_text_by_sentences(
                paragraph, additional_stopwords)
            # tmp = _clean_text_by_sentences(
            #     paragraph, "english")
            if tmp:
                for sent in tmp:
                    sent.paragraph = paragraph_index
                sentences += tmp
                paragraph_index += 1
    # Creates the graph and calculates the similarity coefficient for every pair of nodes.
    graph = _build_graph(
        [sentence.token for sentence in sentences if sentence.token and len(sentence.token) > 2])
    _set_graph_edge_weights(graph)

    # Remove all nodes with all edges weights equal to zero.
    _remove_unreachable_nodes(graph)

    # PageRank cannot be run in an empty graph.
    if len(graph.nodes()) == 0:
        return []

    # Ranks the tokens using the PageRank algorithm. Returns dict of sentence -> score
    pagerank_scores = _pagerank(graph)

    # Adds the summa scores to the sentence objects.
    _add_scores_to_sentences(sentences, pagerank_scores)

    # Sorts the sentences
    sentences.sort(key=lambda s: s.score, reverse=True)
    return sentences, graph, lang


# Main Part
#values = {"text": text4,
#          "n_keywords": "5",
#          "n_sentences": "3",
#          "metricInput": "textrank"
#         }
#sentences, graph, lang = summarize(values['text'])
## 1. Detecting Language
#print("Language dected:", lang)
## 2. Detecting Keywords
#keywords, lemma2words, word_graph, pagerank_scores = keywordsFunc(values#['text'])
#if lang == "en":
#    keyword_formatted = [
#        key + " %.2f (%s)" % (score, ", ".join(lemma2words[key]))
#        for score, key in keywords[:int(values["n_keywords"])]
#            ]
#print("Text keywords are: ")
#print(keyword_formatted) # Text Keywords
## 3. Detecting Sentences
#try:
#    add_alpha(sentences, int(values["n_sentences"]))
#except (ValueError, KeyError):
#    print("Warning: invalid *n* parameter passed!")
#    add_alpha(sentences)
#n_paragraphs = max([x.paragraph for x in sentences]) + 1
#paragraphs = []
#for i in range(n_paragraphs):
#    paragraphs.append(sorted([x for x in sentences if x.paragraph == i], #key=lambda x: x.index))
#res, _, lang = summarize(text4)
#assert lang == "en"
#print('\n')
#print("Text most important sentences with an order:(most important paragraph #is the one with the first sentence) \n")
#for row in res:
#    print(f"{row.score} {row.text} \n")