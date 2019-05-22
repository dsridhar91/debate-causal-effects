import numpy as np
import pandas as pd
import itertools
import os
from scipy import stats
import csv
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
from scipy.spatial.distance import cosine
import dill
import word_category_counter as wc
from sklearn.decomposition import LatentDirichletAllocation
from scipy.special import logit


dat_dir = os.path.join('..', 'dat')
topic_indices = {'abortion':0, 'evolution':1, 'guncontrol':2, 'gaymarriage':3}

def learn_topics(X, K=50):
	lda = LatentDirichletAllocation(n_components=K, learning_method='online', verbose=1)
	print("Fitting", K, "topics...")
	doc_proportions = lda.fit_transform(X)
	score = lda.perplexity(X)
	print("Log likelihood:", score)
	topics = lda.components_
	return score, doc_proportions, topics

def show_topics(vocab, topics, n_words=20):
	topic_keywords = []
	for topic_weights in topics:
		top_keyword_locs = (-topic_weights).argsort()[:n_words]
		topic_keywords.append(vocab.take(top_keyword_locs))

	df_topic_keywords = pd.DataFrame(topic_keywords)
	df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
	df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
	return df_topic_keywords

class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, articles):
		stop = stopwords.words('english')
		return [self.wnl.lemmatize(t) for t in word_tokenize(articles) if t.isalpha() and t not in stop]

def load_document_proportions(embed_dim):
	all_embeddings = {}
	n_topics = embed_dim // 2
	print("N topics:", n_topics)
	for topic in topic_indices.keys():
		embedding_file = os.path.join(dat_dir, topic + '_K=' + str(n_topics), 'doc_proportions.npy')
		id_file = os.path.join(dat_dir, topic, 'ids.npy')
		embedding = np.load(embedding_file)
		ids = np.load(id_file)
		for idx in range(ids.shape[0]):
			all_embeddings[ids[idx]] = embedding[idx, :]
	return all_embeddings

def load_embeddings():
	with open(os.path.join(dat_dir, 'pmfembeddings.dat'), 'rb') as f:
		embeddings = dill.load(f)
	return embeddings


def load_liwc_cat_groups():

	stylistic_acc_categories = ['Articles', 'Certainty', 'Conjunctions','Discrepancy', 'Exclusive',
			'Inclusive', 'Inhibition', 'Negations',  'Prepositions', 'Quantifiers', 'Tentative', 'First Person Singular', 
			'First Person Plural', 'Second Person']

	pos_sentiment_categories = ['Positive Emotion']# , 'Achievement', 'Assent']

	neg_sentiment_categories = ['Negative Emotion', 'Anxiety', 'Anger', 'Sadness'] #, 'Swear Words',]

	categories = {'accomodation':stylistic_acc_categories, 'possent':pos_sentiment_categories, 'negsent':neg_sentiment_categories}
	return categories

	# with open(os.path.join(dat_dir, 'categories.dat'), 'rb') as f:
	# 	categories = dill.load(f)
	# return categories


def compute_outcome(dict1, dict2, categories):
	v1 = get_liwc_vector(dict1, categories)
	v2 = get_liwc_vector(dict2, categories)

	return np.sqrt(np.sum((v1[i]-v2[i])**2 for i in range(v1.shape[0])))

	# if len(categories) < 10:
	# 	return np.sqrt(np.mean(v2) - np.mean(v1))

	# if v1.sum() == 0.0 or v2.sum() == 0.0:
	# 	return 0.0

	# return cosine(v1, v2)

def compute_binary_outcome(dict1, dict2, categories):
	v1 = get_liwc_vector(dict1, categories)
	v2 = get_liwc_vector(dict2, categories)
	sim = 0.0
	if v1.sum() != 0.0 and v2.sum() != 0.0:
		sim = cosine(v1, v2)

	return 1 if sim >= 0.5 else 0



def compute_mean_abs_outcome(dict1, dict2, categories):
	v1 = get_liwc_vector(dict1, categories)
	v2 = get_liwc_vector(dict2, categories)
	return abs(np.mean(v2) - np.mean(v1))



def get_sentiment_score(text, category='possent'):
	lt = LemmaTokenizer()
	tokens = lt(text)
	score = 0.0
	for t in tokens:
		synset = swn.senti_synsets(t)
		if category == 'possent':
			score += np.mean([s.pos_score() for s in swn.senti_synsets(t)])
		else:
			score += np.mean([s.neg_score() for s in swn.senti_synsets(t)])
	return score


def get_liwc_vector(d, categories, use_logit=False):

	vec = np.zeros(len(categories))
	for idx, c in enumerate(categories):
		if use_logit:
			if d[c] > 0:
				vec[idx] = logit(d[c])
			else:
				vec[idx] = d[c]
		else:
			vec[idx] = d[c]
	
	return vec


def load_dicts():
	
	with open(os.path.join(dat_dir, 'posts.dat'), 'rb') as h, open(os.path.join(dat_dir, 'triples.dat'), 'rb') as t:
		
		dposts = dill.load(h)
		triples = dill.load(t)

	return dposts, triples


def poisson_log_ll(corpus, pois):
	
	doc_topic = pois.Et
	topic_term = pois.Eb
	pois_rate = np.dot(doc_topic, topic_term)
	log_rate = np.log(pois_rate)
	counts = np.multiply(corpus, log_rate)
	likelihood = (-1*pois_rate + counts).sum()

	return likelihood


def populate_features_labels(annot_type, embed_dim, use_topic_only, use_accomodation_features=False):

	wc.load_dictionary(wc.default_dictionary_filename())

	discussion_posts, triples = load_dicts()
	post_embeddings = load_document_proportions(embed_dim)
	# post_embeddings = load_embeddings()
	category_types = load_liwc_cat_groups()
	sent_cats = category_types['possent'] + category_types['negsent']

	if use_accomodation_features:
		sent_cats += category_types['accomodation']

	num_topics = len(topic_indices.keys())
	num_triples = len(triples[annot_type])
	dim = embed_dim*num_topics + len(sent_cats)

	if use_topic_only:
		dim = num_topics

	features = np.zeros((num_triples, dim))
	outcome_map = {ct:np.zeros(num_triples) for ct in category_types}
	treatments = np.zeros(num_triples)

	for idx, triple in enumerate(triples[annot_type]):
		p1 = triple[0]
		p2 = triple[1]
		p3 = triple[2]
		annot_val = triple[3]
		did = triple[4]

		treatment = 1 if annot_val > 1 else 0
		topic = discussion_posts[did][p1]['topic']

		embed1 = post_embeddings[p1]
		embed2 = post_embeddings[p2]
		embed = np.hstack([embed1, embed2])

		p1_liwc = wc.score_text(discussion_posts[did][p1]['text'])
		p3_liwc = wc.score_text(discussion_posts[did][p3]['text'])
		p1_sent_vec = get_liwc_vector(p1_liwc, sent_cats)

		tidx = topic_indices[topic]
		if use_topic_only:
			features[idx][tidx] = 1
		else:
			features[idx, tidx*embed_dim:(tidx+1)*embed_dim] = embed
			features[idx, dim - len(sent_cats):] = p1_sent_vec

		treatments[idx] = treatment

		for ct in category_types:
			outcome = compute_outcome(p1_liwc, p3_liwc, category_types[ct])
			outcome_map[ct][idx] = outcome

	return features, treatments, outcome_map


def populate_features_binary_labels(annot_type, embed_dim, use_topic_only):

	wc.load_dictionary(wc.default_dictionary_filename())

	discussion_posts, triples = load_dicts()

	if not use_topic_only:
		post_embeddings = load_document_proportions()
	# post_embeddings = load_embeddings()
	category_types = load_liwc_cat_groups()
	sent_cats = category_types['possent'] + category_types['negsent']

	num_topics = len(topic_indices.keys())
	num_triples = len(triples[annot_type])
	dim = embed_dim*num_topics + len(sent_cats)

	if use_topic_only:
		dim = num_topics

	features = np.zeros((num_triples, dim))
	outcome_map = {ct:np.zeros(num_triples) for ct in category_types}
	treatments = np.zeros(num_triples)

	for idx, triple in enumerate(triples[annot_type]):
		p1 = triple[0]
		p2 = triple[1]
		p3 = triple[2]
		annot_val = triple[3]
		did = triple[4]

		treatment = 1 if annot_val > 1 else 0
		topic = discussion_posts[did][p1]['topic']

		embed1 = post_embeddings[p1]
		embed2 = post_embeddings[p2]
		embed = np.hstack([embed1, embed2])

		p1_liwc = wc.score_text(discussion_posts[did][p1]['text'])
		p3_liwc = wc.score_text(discussion_posts[did][p3]['text'])
		p1_sent_vec = get_liwc_vector(p1_liwc, sent_cats)

		tidx = topic_indices[topic]
		if use_topic_only:
			features[idx][tidx] = 1
		else:
			features[idx, tidx*embed_dim:(tidx+1)*embed_dim] = embed
			features[idx, dim - len(sent_cats):] = p1_sent_vec

		treatments[idx] = treatment

		for ct in category_types:
			outcome = compute_binary_outcome(p1_liwc, p3_liwc, category_types[ct])
			outcome_map[ct][idx] = outcome

	return features, treatments, outcome_map


def tokenize_docs(documents,max_df0=0.80, min_df0=0.02):
	'''
	From a list of documents raw text build a matrix DxV
	D: number of docs
	V: size of the vocabulary, i.e. number of unique terms found in the whole set of docs
	'''
	count_vect = CountVectorizer(tokenizer=LemmaTokenizer(), max_df=max_df0, min_df=min_df0)
	term_counts = count_vect.fit_transform(documents)
	vocab = count_vect.get_feature_names()
	return term_counts,np.array(vocab)



def show_pmf_topics(vocabulary, pmf_model, n_words=20):

	topics = pmf_model.Eb

	for t in range(topics.shape[0]):
		topic_counts = np.array(topics[t,:]).flatten()
		dist = zip(vocabulary, topic_counts)
		top_words = sorted(dist, key=lambda x: x[1])[:n_words]
		print("Top terms in topic", t, ":", ','.join([item[0] for item in top_words]))