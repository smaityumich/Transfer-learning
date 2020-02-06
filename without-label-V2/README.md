# Classifier for target-shift setup
### No labels observed in target distribution (Ref: Lipton et al.)


## Simulation Results
Look at the output.target.shift.nolabel.ipynb for the simulation results

### Simulation procedure steps

- g_0 ~ d-dimensional N(0,1), g_1 ~ d-dimensional N(mu/sqrt(d),1)
- Data generative procedure for H(g_0, g_1, prop):
  - Y ~ Bernoulli(prop)
  - X ~ g_1 if Y = 1; g_0 if Y = 0
- x_source, y_source contains m many data-points from the source distribution P = H(g_0, g_1, prop = 0.5)
- x_target, _ contains n many data-points from the target distribution Q = H(g_0, g_1, prop = prop_target)
- x_test, y_test contains n_test many data-points from the target distribution Q = H(g_0, g_1, prop = prop_target)
- bayes_test are the Bayes decision rules of x_test under Q
- g (the generic classifier used to get the estimate for prop_target: Q(Y=1)) is the kernel smoothed classifier, where the bandwidth of kernel is chosen by 5-fold cross-validation over the parameter grid sequence(0.1,2,by = 0.1)

## Efficiency 
Efficiency of the algorithm is given in cprofile-without-label-classifier.out file

- cProfile package is used to test the efficiency
