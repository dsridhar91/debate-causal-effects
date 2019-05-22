For the **raw** data from the Internet Argument Corpus which we use in this paper,
you'll need to visit this link and download it:
https://nlds.soe.ucsc.edu/iac

Downlaod all contents into this directory. You should have a ./data/ folder
after the download which contains 'fourforums' and 'MechanicalTurk' subdirectories.

To reproduce the experiments as they are in the paper, you only need the processed
and pickled/serialized data from the ./dat/ directory.

To compute the ATE estimates for a reply type:
1) cd src
1) python compute_ate_estimates.py --annot=[reply type] (add --topiconly to run the debate topic only baseline)

The unadjusted baseline estimate is always reported.

To compute the cross validation metrics for a reply type:
1) cd src
2) python cross_validation.py --annot=[reply type] (add --topiconly for the debate topic only baseline)

If you want to re-run LDA with a new number of topics:
1) cd src
2) python fit_lda.py --n_topics=[num topics]
**NOTE: latent topic and document proportions will then be output to ./dat/[debate topic]_N=[num topics]/
You will need to include the --n_topics=[num topics] option to cross val. and ATE estimation
scripts from now on.

Finally, if you downloaded the **raw** data as described above, and you want to 
process the data as we have in the paper:
1) cd src
2) python preprocess.py
**NOTE: please note that your directory structure needs to match exactly
what we had -- read the first two paragraphs.

For questions beyond this, please email: dhanya.sridhar@columbia.edu. 
