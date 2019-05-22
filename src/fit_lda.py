import os
import pandas as pd
from utils import learn_topics, show_topics, tokenize_docs
import argparse
from scipy.sparse import save_npz, load_npz
import dill
import numpy as np

def load_term_counts(path, docs):
    term_count_file = path + '/term_counts.npz'
    vocab_file = path + '/vocab.npy'
    if os.path.exists(term_count_file):
        return load_npz(term_count_file), np.load(vocab_file)
    else:
        term_counts, vocab = tokenize_docs(docs)
        return term_counts, vocab

def get_posts(topic, discussion_posts, discussion_topics):
    print("Working on topic", topic)
    ids = []
    docs = []
    topic_discussions = discussion_topics[discussion_topics.topic == topic]['discussion_id']

    for did in topic_discussions.values:
        if did in discussion_posts:
            for pid in discussion_posts[did]:
                ids.append(pid)
                docs.append(discussion_posts[did][pid]['text'])

    return ids, docs

def load_discussions():
    discussion_topics = pd.read_csv(topics_file)
    discussion_topics['topic'] = discussion_topics['topic'].str.replace(' ', '').str.replace('"','')
    discussion_topics = discussion_topics[discussion_topics.topic.isin(val_topics)]

    with open(os.path.join('..', 'dat', 'posts.dat'), 'rb') as h:
        discussion_posts = dill.load(h)

    return discussion_topics, discussion_posts

def main():
    discussion_topics, discussion_posts = load_discussions()
    for topic in val_topics:
        outdir = os.path.join('..', 'dat', topic)
        ids, docs = get_posts(topic, discussion_posts, discussion_topics)
        term_counts, vocab = load_term_counts(outdir, docs)
        score, doc_proportions, topics = learn_topics(term_counts, K=n_topics)
        print(show_topics(vocab, topics))
        
        topic_dir = os.path.join('..', 'dat', topic + '_K=' + str(n_topics))
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(topic_dir, exist_ok=True)
        save_npz(os.path.join(outdir, 'term_counts'), term_counts)
        np.save(os.path.join(topic_dir, 'doc_proportions'), doc_proportions)
        np.save(os.path.join(outdir, 'vocab'), vocab)
        np.save(os.path.join(topic_dir, 'topics'), topics)
        np.save(os.path.join(outdir, 'ids'), np.array(ids))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-topics", action="store", default=20)
    args = parser.parse_args()
    n_topics = int(args.n_topics)
    val_topics = {'evolution', 'abortion', 'guncontrol', 'gaymarriage'}
    topics_file = os.path.join('..', 'data', 'fourforums', 'annotations','topic.csv')

    main()
