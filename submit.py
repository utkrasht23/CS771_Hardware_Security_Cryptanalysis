import numpy as np
import sklearn

# You are allowed to import any submodules of numpy or sklearn e.g. sklearn.metrics.accuracy_score to calculate accuracy of a learnt model
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_map, my_params etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here

################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to map raw features to proper embeddings
	# Your embeddings will be used to train a linear SVM model
	
	# Transform the challenge bits from {0, 1} to {-1, 1}.
	# This centers the data and dramatically improves the convergence speed of the LinearSVC.
	X_sym = 1 - 2 * X
	
	# Extract the even-indexed bits (16 features)
	# E shape: (N, 16)
	E = X_sym[:, 0::2]
	
	# Extract the odd-indexed bits (16 features)
	# O shape: (N, 16)
	O = X_sym[:, 1::2]
	
	# Compute all pair-wise products between even and odd bits (16 x 16 = 256 features).
	# This represents the c_i * c_j terms created when calculating he * ho.
	# We use numpy broadcasting to compute this efficiently for all samples at once.
	# cross shape: (N, 256)
	cross = (E[:, :, np.newaxis] * O[:, np.newaxis, :]).reshape(X.shape[0], -1)
	
	# The final mapped feature vector has a dimensionality of D = 16 + 16 + 256 = 288.
	# This dimensionality is very small, ensuring you don't lose marks for excessively large maps.
	X_map = np.hstack((E, O, cross))
	
	return X_map

################################
# Non Editable Region Starting #
################################
def my_params( X_map, X_raw, y ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to return your preferred hyperparameters
	# The original and mapped train features and train labels have been supplied in case you wish to perform hyperparameter turning via validation.
	# You may ignore the inputs and provide fixed values of the hyperparameters that you found to work well on the public train/test set.
	# You need not set every single hyperparameter. The ones you do not set will be set to their default value. Please refer to the API for sklearn.svm.LinearSVC for the default values.
	
	# Since N > D (e.g., 7500 > 288), setting dual=False strictly solves the primal 
	# SVM problem which runs much faster than the dual problem. 
	# A slightly elevated C ensures maximum accuracy (zero bias) on separable PUF data.
	my_params = {
		"penalty": "l2",
		"loss": "squared_hinge",
		"C": 1.0,
		"tol": 1e-4,
		"max_iter": 5000,
		"dual": False
	}
	
	return my_params