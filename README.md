# ggp-poisson-example
Example of GLMM for Poisson likelihood.

1. Install limix-inference
```bash
conda install -c conda-forge limix-inference
```

2. Clone this repository
```bash
git clone https://github.com/glimix/ggp-poisson-example.git
```

3. Create and run the example
```bash
cd ggp-poisson-example
python create_example.py
python run_example.py
```

That is it.
You should see something similar to
```
--- Model before optimization ---
LML: -493.26572309
Fixed-effect sizes: [ 0.  0.  0.]
Variances:
  covariance0: 1.00000000
  covariance1: 1.00000000
  covariance2: 1.00000000
  Eye        : 1.00000000

...

--- Model after optimization ---
LML: -406.67731671
Fixed-effect sizes: [ 0.35329736  0.13210078  0.19191181]
Variances:
  covariance0: 0.08680102
  covariance1: 0.00266242
  covariance2: 0.09787977
  Eye        : 0.01149269
```

The numbers don't need to match as the example itself
is stochastic.
