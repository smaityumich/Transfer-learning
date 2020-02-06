# Classifier for target-shift setup
### No labels observed in target distribution (Ref: Lipton et al.)


## Simulation Results
Look at the output.target.shift.nolabel.ipynb for the simulation results

### Simulation procedure steps

- g_0 ~ d-dimensional N(0,I), g_1 ~ d-dimensional N(mu/sqrt(d),1)
- Data generative procedure for P:
  - Y ~ Bernoulli(prop_source)
  - X ~ g_1 if Y = 1; g_0 if Y = 0
  - (X,Y) ~ P
- Data generative procedure for Q is similar, but with prop_target
- x_source, y_source contains m many data-points from the source distribution P = P(g_0, g_1, prop_source = 0.5)



## Efficiency 
Efficiency of the algorithm is given in cprofile-without-label-classifier.out file

- cProfile package is used to test the efficiency
