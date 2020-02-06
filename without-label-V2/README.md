# Classifier for target-shift setup
### No labels observed in target distribution (Ref: Lipton et al.)
!(meme.jpg)

## Simulation Results
Look at the output.target.shift.nolabel.ipynb for the simulation results

### Simulation procedure steps

- g_0 ~ d-dimensional N(0,1), g_1 ~ d-dimensional N(distance/sqrt(d),1)
- Data generative procedure for H(g_0, g_1, prop):
  - Y ~ Bernoulli(prop)
  - X ~ g_1 if Y = 1; g_0 if Y = 0
- x_source, y_source contains m many data-points from the source distribution P = H(g_0, g_1, prop = 0.5)
- x_target, _ contains n many data-points from the target distribution Q = H(g_0, g_1, prop = prop_target)
- x_test, y_test contains n_test many data-points from the target distribution Q = H(g_0, g_1, prop = prop_target)
- bayes_test are the Bayes decision rules of x_test under Q
- g (the generic classifier used to get the estimate for prop_target: Q(Y=1)) is the kernel smoothed classifier, where the bandwidth of kernel is chosen by 5-fold cross-validation over the parameter grid sequence(from = 0.1, to = 2, by = 0.1)
- x_test is used to get predicted labels (data-points independent of x_target)
- prediction-error and bayes-error are calculated; along with excess-risk = prediction-error - bayes-error
- Each experiment with a particular setup is repeated 100 times to get CI for error
- Parameter setup
  - m = [25, 50, 100, 200, 400, 800, 1600, 3200]
  - n = 500
  - n_test = 1000
  - d = 5
  - prop_target = 0.8
  - distance = 0.5
- Plots for excess-risk CI, and CI for estimation error of prop_target are given  



## Efficiency

Efficiency test of the algorithm is given in cprofile-without-label-classifier.out file

- cProfile module is used to test the efficiency
- Majority of the time is spent on calculating kernel density at a point 
  - In the method sklearn.neighbors.kd_tree.BinaryTree.kernel_density; about 0.023 seconds in ARM v8 model CPU 
