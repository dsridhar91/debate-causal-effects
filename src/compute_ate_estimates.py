import utils as ut
import numpy as np
import word_category_counter as wc
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from scipy.special import logit
from scipy.stats import ttest_ind, ttest_rel
from sklearn.metrics import f1_score
from ate import ate_estimates
import sys

def main():
	features, treatment_labels, outcome_map = ut.populate_features_labels(annot_type, embed_dim, use_topic_only, use_accomodation_features=True)

	logistic = LogisticRegression(solver='liblinear', penalty='l1')
	logistic.fit(features, treatment_labels)
	print(logistic.score(features, treatment_labels))
	treat_prob = logistic.predict_proba(features)[:, 1]

	for ct in outcome_map.keys():
		print("Working on outcome type:", ct)
		outcomes = outcome_map[ct]
		outcomes_st_treated = outcomes[treatment_labels==1]
		features_st_treated = features[treatment_labels==1]
		outcomes_st_not_treated = outcomes[treatment_labels==0]
		features_st_not_treated = features[treatment_labels==0]

		model_st_treated = Ridge()
		model_st_not_treated = Ridge()
		model_st_treated.fit(features_st_treated, outcomes_st_treated)
		model_st_not_treated.fit(features_st_not_treated, outcomes_st_not_treated)

		expected_outcomes_st_treated = model_st_treated.predict(features)
		expected_outcomes_st_not_treated = model_st_not_treated.predict(features)

		estimates = ate_estimates(expected_outcomes_st_not_treated, expected_outcomes_st_treated, 
			treat_prob, treatment_labels, outcomes, truncate_level=0.03)
		print(estimates)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--topiconly", action="store_true")
	parser.add_argument("--annot", action="store", default='NN')
	parser.add_argument('--n-topics', action="store", default=50)
	args = parser.parse_args()

	use_topic_only = args.topiconly
	annot_type = args.annot

	print("Found options for topic only: %s, annot: %s" % (use_topic_only,annot_type))

	topic_indices = {'abortion':0, 'evolution':1, 'guncontrol':2, 'gaymarriage':3}
	embed_dim = int(args.n_topics) * 2
	num_topics = len(topic_indices.keys())

	wc.load_dictionary(wc.default_dictionary_filename())

	main()
	
