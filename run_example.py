from numpy import arange, loadtxt

from limix_inference.cov import EyeCov, GivenCov, SumCov
from limix_inference.mean import LinearMean
from limix_inference.ggp import ExpFamGP

def create_mean(X):
    mean = LinearMean(X.shape[1])
    mean.set_data((X, ))
    return mean

def create_covariances(covariances):
    n = None
    covs = dict()
    for k, K in iter(covariances.items()):
        covs[k] = GivenCov(K)
        n = K.shape[0]
        covs[k].set_data((arange(n), arange(n)))

    covs['eye'] = EyeCov()
    covs['eye'].set_data((arange(n), arange(n)))

    return covs

y = loadtxt('pheno.csv', delimiter=',')

mean = create_mean(loadtxt('covariates.csv', delimiter=','))

covs = dict()
for i in range(3):
    prefix = 'covariance%d' % i
    covs[prefix] = loadtxt(prefix + '.csv', delimiter=',')
covs = create_covariances(covs)

cov = SumCov(list(covs.values()))

ggp = ExpFamGP(y, 'poisson', mean, cov)

def print_info(ggp, mean, covs):
    print('LML: %.8f' % ggp.feed().value())
    print('Fixed-effect sizes:', mean.effsizes)
    print('Variances:')
    for i in range(3):
        prefix = 'covariance%d' % i
        print('  %s: %.8f' % (prefix, covs[prefix].scale))
    print('  Eye        : %.8f' % covs['eye'].scale)
    print()

print('--- Model before optimization ---')
print_info(ggp, mean, covs)
ggp.feed().maximize(progress=True)

print()
print('--- Model after optimization ---')
print_info(ggp, mean, covs)

