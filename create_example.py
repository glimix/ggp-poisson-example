from numpy import arange, dot, ones, append, savetxt, zeros, asarray, exp
from numpy.random import RandomState


def create_phenotype(covariates, covariances):
    nsamples = covariates.shape[0]
    y = zeros(nsamples)

    random = RandomState(0)

    y += dot(covariates, random.randn(covariates.shape[1]))
    for K in covariances:
        y += random.multivariate_normal(zeros(nsamples), K)

    y /= 10.

    return asarray([random.poisson(exp(yi)) for yi in y])


def create_covariance(n, seed):
    random = RandomState(seed)
    X = random.randn(n, n + 1)
    K = dot(X, X.T)
    K /= K.diagonal().mean()
    return K

def create_covariates(n, nc):
    random = RandomState(0)
    covariates = ones((n, 1))
    covariates = append(covariates, random.randn(n, nc),
                        axis=1)
    return covariates

n = 100


covariates = create_covariates(n, 2)

K0 = create_covariance(n, 0)
K1 = create_covariance(n, 1)
K2 = create_covariance(n, 2)

y = create_phenotype(covariates, [K0, K1, K2])

savetxt('pheno.csv', y, delimiter=',', fmt='%.4e')
savetxt('covariates.csv', covariates, delimiter=',', fmt='%.4e')
savetxt('covariance0.csv', K0, delimiter=',', fmt='%.4e')
savetxt('covariance1.csv', K1, delimiter=',', fmt='%.4e')
savetxt('covariance2.csv', K2, delimiter=',', fmt='%.4e')
