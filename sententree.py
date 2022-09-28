import os
import pandas as pd
import random

from pathlib import Path
import copy
import math

import nltk
from nltk import word_tokenize
from nltk import pos_tag

import networkx as nx
from treelib import Tree
from graphviz import Digraph

from prefixspan import PrefixSpan
import patterns
from collections import Counter


def tag_words(sents):
    """
    tag_words: adds the occurence to a word for each word in a sublist

    parameters:
        sents: list of lists, where a sublist represents a sentence
    
    returns: a list of lists, where elements of a sublist are tagged by their occurence
    """ 
    for j in range(len(sents)):
        sent = sents[j]
        d = {}
        for i in range(len(sent)):
            # Check if the word is in dictionary
            word = sent[i]
            if word in d:
                # Increment count of word by 1
                d[word] = d[word] + 1
            else:
                # Add the word to dictionary with count 1
                d[word] = 1
            word = word+"_"+str(d[word])
            sent[i]= word
    return sents



def is_sublist(lst1, lst2):
    """
    is_sublist: true if lst1 is sublist of lst2, else false
    """ 
    return set(lst1) <= set(lst2)

def is_super_pattern(ref_pat):
    """
    is_super_function: checks if pat is a super_pattern of ref_pat
    parameters:
        ref_pat: (list) an encoded pattern representing a sentence
    returns: True/False
    """ 
    target_len = len(ref_pat) + 1

    def _inner(pat, support):
        if len(pat) != target_len:
            return False
        if not is_sublist(ref_pat, pat):
            return False
        return True
        
    return _inner

def divide_sentences(sentences, cur_pat, min_support):
    """
    divide_sentences: splits sentences into sentences that match the pattern, do not match pattern
    parameters:
        sentences: list of lists, where a sublist represents a sentence
        cur_pat: list of encoded words
        min_support (int)
    returns: returns list of tuples, each tuple has sentences matching pattern, not matching pattern
    """ 
    miner = PrefixSpan(sentences)
    options = miner.frequent(min_support, filter=is_super_pattern(cur_pat))
    if len(options) == 0:
        return []
    
    best_support, best_option = sorted(options, key=lambda e: e[0])[-1]
    has_pattern = [s for s in sentences if is_sublist(best_option, s)]
    residue = [s for s in sentences if s not in has_pattern]
    
    return [(has_pattern, best_option)] + divide_sentences(residue, cur_pat, min_support)
    
    
def find_leaf_patterns(sentences, cur_pat, *, min_support=1):
    """
    find_leaf_patterns: grow current pattern until reaching min_support
    parameters:
        sentences: list of lists, where a sublist represents a sentence
        cur_pat: (list) encoded words
        min_support (int)
    returns: tree structure describing how the pattern grew at each level
    """ 
    def pattern_added(entry):
        entry = entry.copy()
        added =  list(set(Counter(entry["pattern"]).items()) - set(Counter(cur_pat).items()))
        entry["delta"] = added[0][0]
        return entry
    
    child_groups = divide_sentences(sentences, cur_pat, min_support)
    children = [find_leaf_patterns(s, p, min_support=min_support) for s,p in child_groups]
    children = [pattern_added(c) for c in children]  
    return {"delta": None, 
            "pattern": cur_pat, 
            "size": len(sentences), 
            "children": children}



def pretty(list_tree, encoder):
    """
    pretty: build tree from tree stored as a list
    parameters:
        list_tree: (list) tree structure
        encoder: used to decode words
    returns: tree made up of leaf patterns
    """     
    t = Tree()
    decoded = "" if list_tree["delta"] in [[], None] else encoder.decode(value=list_tree["delta"])
    t.create_node(tag = f"{decoded} ({list_tree['size']})", data = list_tree["pattern"])
    
    children = [pretty(child, encoder) for child in list_tree["children"]]
    for child in children:
        t.paste(t.root, child)
    return t



def get_delta_index(child):
    """
    get_delta_index: finds and returns index of delta in child's pattern
    """
    child_p = child["pattern"]
    for j in range(len(child_p)):
        if child_p[j] == child["delta"]:
            return j

def rename_child(child, new_delta, index, patterns, encoder):
    """
    rename_child: update child delta, pattern and children information
    parameters:
        child: (dict)
        new_delta: (str) new delta value
        index: (int) where to insert new delta into child pattern
    returns: updated child
    """ 
    delta = [*patterns.encode([new_delta], encoder=encoder)][0]
    child["pattern"][index] = delta
    child["delta"] = delta
    child["children"] = []
    return child

def build_summary(children, k, encoder): 
    """
    build_summary: create elements of summary describing the list of children
    parameters:
        children: list of dicts, each element stores information about a child
        k: (int) number of values in final summary
        encoder: used to decode words
    returns: (list) or None 
    """ 
    deltas = [encoder.decode(child["delta"]) for child in children]
    tags = pos_tag(deltas)
    nums = [tag[0] for tag in tags if tag[1] == "CD"]
    words = [tag[0] for tag in tags if tag[1] != "CD"]
    num_vals_left = str(len(nums) + len(words) - k)
    if len(nums) > 0 and len(words) > 0:
        return [nums, words, num_vals_left + "+ words and numbers"]
    if len(words) > 0:
        return [words, num_vals_left + "+"]
    if len(nums) > 0:
        return [nums, num_vals_left + "+"]
    return None

def build_exemplar_summary(children, encoder, patterns, k=3):
    """
    build_exemplar_summary: update leaf patterns with summary information
    parameters:
        children: list of dicts, each element stores information about a child
        encoder: used to decode words
        patterns: used to encode words
        k: (int) number of values in final summary
    returns: (list of dicts) summarizing children 
    """ 
    if k < 1:
        return []
    
    # Get summary values
    if k > len(children) - 1: k = len(children) - 2
    summary = build_summary(children, k-1, encoder)
    if len(summary) > 2:
        samples = random.sample(summary[0], (k-1)/2) + random.sample(summary[1], (k-1) - ((k-1)/2))
    else:
        samples = random.sample(summary[0], k-1)
        
    # Create summary children
    children_new = []
    index = get_delta_index(children[0])
    for new_node in samples:
        children_new.append(rename_child(copy.deepcopy(children[0]), str(new_node), index, patterns, encoder))        
    children_new.append(rename_child(children[k], summary[-1], index, patterns, encoder))
    
    return children_new   

def prune_tree(tree, encoder, patterns, min_support=1):
    """
    prune_tree: remove children below min support and replace with children's summary for level = min_support - 1
    parameters:
        tree: list of dicts, a tree structure describing how sequential patterns grew
        encoder: used to decode words
        patterns: used to encode words
        min_support: (int)
    returns: (list of dicts) new tree with summary of children 
    """ 
    for child in tree["children"]:
        if child["size"] >=  min_support:
            prune_tree(child, encoder, patterns, min_support = min_support)
            
    # Summarize children with support < min_support, add summary to parent pattern
    children_below_support = [child for child in tree["children"] if child["size"] <  min_support]
    if children_below_support !=[]:
        tree["children"] = build_exemplar_summary(children_below_support, encoder, patterns)
    if tree["children"] == []:
        decoded = [encoder.decode(value=word) for word in tree["pattern"]]
    return tree


def make_graph_from_leaves(tree, G, encoder, patterns, min_support=1):
    """
    make_graph_from_leaves: Finds leaf pattern and adds it to the tree
    parameters:
        tree: (list of dicts) a tree structure describing how sequential patterns grew
        G: (graph) of sequential leaf patterns
        encoder: used to decode words
        patterns: used to encode words
        min_support: (int)
    returns: (graph) of all leaf patterns
    """ 
    for child in tree["children"]:
        make_graph_from_leaves(child, G, encoder, patterns, min_support = min_support)
        
    if tree["children"] == []:
        decoded = [encoder.decode(value=word) for word in tree["pattern"]]
        for word in decoded:
            if G.has_node(word):
                G.nodes[word]["count"] += tree["size"]
            else:
                G.add_node(word, count=tree["size"])
                    
        for source, target in zip(decoded, decoded[1:]):
            G.add_edge(source, target)
            
def simplify_graph(G):
    """
    simplify_graph: Merges nodes with the same support
    parameters:
        G: (graph) of sequential leaf patterns
    returns: (graph) of all leaf patterns with merged nodes
    """ 
    for node in G.nodes():
        children = [*G.out_edges(nbunch=[node])]
        if len(children) == 1 and G.nodes[node]["count"] == G.nodes[children[0][1]]["count"]:
            child = children[0][1]
            combined_name = node +" "+ child
            
            mapping = {child: combined_name}
            G = nx.relabel_nodes(G, mapping)
            [G.add_edge(parent, combined_name) for parent in G.predecessors(node)]
            G.remove_node(node)
            
            G = simplify_graph(G)
            return G
    return G




def get_label(tagged_label):
    """
    get_label: remove the occurrence tag from node label
    parameters:
        tagged_label: (string) label with occurrence tag
    returns: label without occurrence tag
    """ 
    split_str = tagged_label.split()
    for i, word in enumerate(split_str):
        head, sep, tail = word.partition('_')
        split_str[i] = head
    return " ".join(split_str)

def get_font_info(support, sents_count):
    """
    get_font_info: determine font size and color based on support for sequential pattern
    parameters:
        support: (int) support of a pattern
    returns: 
        size: (str)
        color: (str)
    """ 
    size = math.sqrt(support) * 3 + 8
    if size > sents_count/4 + 8:
        size = sents_count/4 + 8
        color="#3182bd"
    else:
        color = "#9ecae1"
    return str(size), color
                
def change_node_data(tagged_label, node_data, sents_count, g):
    """
    change_node_data: Gets node attribute data, adds node to graphviz graph
    parameters:
        tagged_label: (str) label with occurrence tag
        node_data: (list of dicts) node attribute data 
        sents_count: number of sentences in cluster
        g: (graph) Graphviz graph of sequential patterns
    """ 
    label = get_label(tagged_label)
    font_size, color =  get_font_info(node_data[tagged_label]["count"], sents_count)
    g.node(tagged_label, label = label, fontsize = font_size, fontcolor=color, shape="plaintext", fontname="Arial")

def make_graphviz(G, g, sents_count):
    """
    make_graphviz: Create graphviz graph from networkx graph
    parameters:
        g: (graph) Graphviz graph of sequential patterns
        G: (Graph) networkx graph of sequential patterns
        sents_count: number of sentences in cluster
    """ 
    node_data = G.nodes(data=True)

    for (s, t) in G.edges:
        g.edge(s,t)
        change_node_data(s, node_data, sents_count, g)
        change_node_data(t, node_data, sents_count, g)

def build_sententree(filepath, summary_len=3, min_support=2):
    cluster_df = pd.read_csv(filepath, index_col=[0])
    a_cluster = cluster_df.groupby("cluster").get_group(0)
    split_sents = [sent.split() for sent in a_cluster.sentences]
    tagged_split_sents = tag_words(split_sents)
    
    encoder = patterns.Encoder()
    encoded_sents = [patterns.encode(sent, encoder=encoder) for sent in tagged_split_sents]
    
    pattern_tree = find_leaf_patterns(encoded_sents, [], min_support=1)
    t = pretty(pattern_tree, encoder)
    pruned_tree = prune_tree(copy.deepcopy(pattern_tree), encoder, patterns, min_support)

    G = nx.DiGraph()
    make_graph_from_leaves(pruned_tree, G, encoder, patterns, min_support)
    G = simplify_graph(G)
    
    g = Digraph(strict=True)
    g.graph_attr["rankdir"] = "LR"
    make_graphviz(G, g, len(encoded_sents))
    return g


        

        
        