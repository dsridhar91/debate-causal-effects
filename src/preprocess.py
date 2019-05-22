#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
from importlib import reload
import pmf
import utils as ut
import json
import dill

val_topics = {'evolution', 'abortion', 'guncontrol', 'gaymarriage'}

columns = ['key', 'nicenasty', 'questioning-asserting', 'attack', 'fact-feeling']

info_columns = ['discussion_id', 'response_post_id', 'quote_post_id', 'quote', 'response'] + columns[1:]

qr_file = os.path.join('..', 'data', 'fourforums', 'annotations', 'mechanical_turk', 'qr_averages.csv')
metadata_file = os.path.join('..', 'data', 'fourforums', 'annotations', 'mechanical_turk', 'qr_meta.csv')
topics_file = os.path.join('..', 'data', 'fourforums', 'annotations','topic.csv')

'''
Filter the discussions in valid topics and clean topic strings
'''
print("Extracting discussions from valid topics..")

discussion_topics = pd.read_csv(topics_file)
discussion_topics['topic'] = discussion_topics['topic'].str.replace(' ', '').str.replace('"','')
discussion_topics = discussion_topics[discussion_topics.topic.isin(val_topics)]
dt = {d[0]:d[1] for d in discussion_topics.values}

'''
Merge the annotated QR pairs with metadata
Keep only 4 annotation types
Quote, response, post ids -- these match correctly with post metadata (i.e., parent id)
'''

print("Merging QR annotations with text/id metadata..")

qr = pd.read_csv(qr_file)
md = pd.read_csv(metadata_file)

qr_sub = qr[columns]
qr_md = qr_sub.merge(md, how='inner', on='key')
qr_md = qr_md[~qr_md.quote_post_id.isnull() & ~qr_md.response_post_id.isnull()]
qr_md = qr_md[qr_md.discussion_id.isin(dt)]
qr_pairs = qr_md[info_columns]

'''
For discussions that have been annotated, store post text, id, parent post id, author
Copy over topic from discussion's topic
E.g. To get text of post 4 in discussion 30: discussion_posts[30][4]['text']
'''

print("Extracting posts from discussions..")

discussions = qr_pairs['discussion_id'].values
post_info_mapping = {2:'author', 3:'text', 5:'parent_id'}
discussions_dir = os.path.join('..', 'data', 'fourforums', 'discussions')
discussion_posts = {}

for d in discussions:
    postdict = {}
    fname = os.path.join(discussions_dir, str(d) + '.json')
    f = open(fname)
    (posts, annotations, metadata) = json.load(f)
    for p in posts:
        post = {}
        post['topic'] = dt[d]
        for index,info in post_info_mapping.items():
            post[info] = p[index]
        pid = p[0]
        postdict[pid] = post
    discussion_posts[d] = postdict  

'''
For annotated QR pairs, remove quote from response text
Store the annotation info, i.e. value for each annot. type if exists
For (quote post id, response id) keys, store as above
'''

print("Extracting QR pairs and annotation info..")

annotation_mappings = {5:'NN', 6:'QA', 7:'NA', 8:'FF'}
discussion_qr_annots = {}

for entry in qr_pairs.values:
    did = entry[0]
    rpid = entry[1]
    qpid = entry[2]
    quote = entry[3]
    response = entry[4]
    
    discussion_posts[did][rpid]['text'] = discussion_posts[did][rpid]['text'].replace(quote, '')
    
    key = (qpid, rpid)
    annot_info = {}
    for index, annot_type in annotation_mappings.items():
        if entry[index] != float('nan'):
            if abs(entry[index]) > 1:
                annot_info[annot_type] = entry[index]

    if did in discussion_qr_annots:
        discussion_qr_annots[did][key] = annot_info
    else:
        discussion_qr_annots[did] = {key:annot_info}          

'''
Some posts still have quotations from other posts.
Locate these posts; iterate over each sentence from
other posts in discussion and remove if quoted.
'''

print("Cleaning quoted text from posts..")

posts_with_quotes = []
for did, posts in discussion_posts.items():
    for pid, p in posts.items():
        if 'originally posted by' in p['text'].lower():
            posts_with_quotes.append((did, pid))


for (did, pid) in posts_with_quotes:
    
    text = discussion_posts[did][pid]['text']
    
    for other_pid in discussion_posts[did]:
        if pid == other_pid:
            continue
        other_text = discussion_posts[did][other_pid]['text'].split('\n')

        for line in other_text:
            if len(line) > 5:
                if line in text:
                    text = text.replace(line, "")

    discussion_posts[did][pid]['text'] = text

'''
Compute triples for each annot. type
The third post in triple must satisfy:
- author is same as post 1 (i.e., quote post)
- parent is post 2 (i.e., response post)
- on order of 1000s of triples per annot. type
'''

print("Computing triples per annotation type..")

annotation_types = ['NN', 'QA', 'NA', 'FF']
triples = {a:[] for a in annotation_types}

for did, qrinfo in discussion_qr_annots.items():
    for (qid, rid), annot_info in qrinfo.items():
        author = discussion_posts[did][qid]['author']
        
        #search for the third post in triple
        candidate_child_posts = [pid for pid in discussion_posts[did] if discussion_posts[did][pid]['parent_id']==int(rid)]
        valid_children = [p for p in candidate_child_posts if discussion_posts[did][p]['author'] == author]
        for vc in valid_children:
            for a in annotation_types:
                if a in annot_info:
                    triples[a].append((qid, rid, vc, annot_info[a], did))


print("Lengths of dis. posts, dis. annots and triples: %d; %d; %d" % (len(discussion_posts), len(discussion_qr_annots), len(triples)))

'''
Pickle dictionaries
''' 
print("Picking dictionaries...")

dat_dir = os.path.join('..', 'dat')

with open(os.path.join(dat_dir, 'posts.dat'), 'wb') as h, open(os.path.join(dat_dir, 'annots.dat'), 'wb') as s, open(os.path.join(dat_dir, 'triples.dat'), 'wb') as t:
    
    dill.dump(discussion_posts, h)
    dill.dump(discussion_qr_annots, s)
    dill.dump(triples, t)

