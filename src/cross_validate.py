import utils as ut
import numpy as np
import word_category_counter as wc
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from scipy.special import logit
from scipy.stats import ttest_ind, ttest_rel
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score
from ate import ate_estimates
import sys

def main():
	if bin_outcome:
		features, treatment_labels, outcome_map = ut.populate_features_binary_labels(annot_type, embed_dim, use_topic_only)
	else:
		features, treatment_labels, outcome_map = ut.populate_features_labels(annot_type, embed_dim, use_topic_only, use_accomodation_features=True)
	folds = 5
	splits = np.random.randint(0, folds+1, size=features.shape[0])

	f1 = np.zeros(folds)
	mse_treated = np.zeros((folds, len(outcome_map.keys())))
	mse_not_treated = np.zeros((folds, len(outcome_map.keys())))
	for k in range(folds):
		idx = splits != k
		val_idx = splits == k
		tr_features = features[idx]
		tr_labels = treatment_labels[idx]
		te_features = features[val_idx]
		te_labels = treatment_labels[val_idx]

		logistic = LogisticRegression(solver='liblinear', penalty='l1')
		logistic.fit(tr_features, tr_labels)
		pred = logistic.predict(te_features)
		f1[k] = f1_score(te_labels, pred)
		
		for c_idx, ct in enumerate(outcome_map.keys()):
			print("Working on outcome type:", ct)
			te_outcomes = outcome_map[ct][val_idx]
			tr_outcomes = outcome_map[ct][idx]

			if bin_outcome:
				if tr_outcomes.sum() == 0.0:
					continue

			outcomes_st_treated = tr_outcomes[tr_labels==1]
			features_st_treated = tr_features[tr_labels==1]
			outcomes_st_not_treated = tr_outcomes[tr_labels==0]
			features_st_not_treated = tr_features[tr_labels==0]
			
			if bin_outcome:
				model_st_treated = LogisticRegression(solver='liblinear', penalty='l1')
				model_st_not_treated = LogisticRegression(solver='liblinear', penalty='l1')
			else:
				model_st_treated = Ridge()
				model_st_not_treated = Ridge()
			
			model_st_treated.fit(features_st_treated, outcomes_st_treated)
			model_st_not_treated.fit(features_st_not_treated, outcomes_st_not_treated)

			te_outcomes_st_treated = te_outcomes[te_labels==1]
			te_outcomes_st_not_treated = te_outcomes[te_labels==0]
			te_features_st_treated = te_features[te_labels==1]
			te_features_st_not_treated = te_features[te_labels==0]

			if bin_outcome:
				predict_treated = model_st_treated.predict_proba(te_features_st_treated)[:,1]
				predict_not_treated = model_st_not_treated.predict_proba(te_features_st_not_treated)[:,1]
				mse_treated[k][c_idx] = f1_score(te_outcomes_st_treated, model_st_treated.predict(te_features_st_treated))
				mse_not_treated[k][c_idx] = f1_score(te_outcomes_st_not_treated, model_st_not_treated.predict(te_features_st_not_treated))

			else:
				predict_treated = model_st_treated.predict(te_features_st_treated)
				predict_not_treated = model_st_not_treated.predict(te_features_st_not_treated)
				mse_treated[k][c_idx] = np.sqrt(mean_squared_error(te_outcomes_st_treated, predict_treated))
				mse_not_treated[k][c_idx] = np.sqrt(mean_squared_error(te_outcomes_st_not_treated, predict_not_treated))

	print("Mean F1 predicting treatment (and std.):", f1.mean(), f1.std())
	print("Mean MSE Q0 for accomodation, pos. sent. and neg. sent. (and std, respectively):", mse_not_treated.mean(axis=0), mse_not_treated.std(axis=0))
	print("Mean MSE Q1 or accomodation, pos. sent. and neg. sent. (and std, respectively):", mse_treated.mean(axis=0), mse_treated.std(axis=0))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--topiconly", action="store_true")
	parser.add_argument("--annot", action="store", default='NN')
	parser.add_argument('--n-topics', action="store", default=50)
	parser.add_argument('--bin-outcome', action="store_true")
	args = parser.parse_args()

	use_topic_only = args.topiconly
	annot_type = args.annot
	bin_outcome = args.bin_outcome

	print("Found options for topic only: %s, annot: %s" % (use_topic_only,annot_type))

	topic_indices = {'abortion':0, 'evolution':1, 'guncontrol':2, 'gaymarriage':3}
	embed_dim = int(args.n_topics) * 2
	num_topics = len(topic_indices.keys())

	wc.load_dictionary(wc.default_dictionary_filename())

	main()
	
