# Paradigm Free Mapping vs Total Activation

This repository contains a Jupyter Notebook that aims to compare the Paradigm Free Mapping (PFM) and Total Activation (TA) methods. For a given input dataset, it compares the two methods in the following way based on the regularization parameter selection approach:

1. Computing the regularization path for both methods and plotting the BIC and AIC selected regularization parameters in the path.
2. Selecting the standard deviation of the estimation of the noise of the original signal as the regularization parameter. This parameter is then updated on every iteration until the standard deviation of the residuals converges to that of the noise estimate with a given precision.

The notebook also generates informative plots for each of the conditions.
