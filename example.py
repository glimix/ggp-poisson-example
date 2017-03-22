from numpy import arange, sqrt
from numpy.random import RandomState

from limix_inference.cov import EyeCov, LinearCov, SumCov
from limix_inference.mean import OffsetMean
from limix_inference.example import offset_mean
from limix_inference.example import linear_eye_cov
from limix_inference.ggp import ExpFamGP
from limix_inference.lik import PoissonProdLik
from limix_inference.link import LogitLink
from limix_inference.random import GGPSampler


def create_mean(N):
    mean = OffsetMean()
    mean.offset = 1.0
    mean.set_data(arange(N), purpose='sample')
    mean.set_data(arange(N), purpose='learn')
    return mean

def create_cov(N):
    random = RandomState(0)
    X = random.randn(N, N + 1)
    X -= X.mean(0)
    X /= X.std(0)
    X /= sqrt(X.shape[1])

    cov_left = LinearCov()
    cov_left.scale = 1.0
    cov_left.set_data((X, X), purpose='sample')
    cov_left.set_data((X, X), purpose='learn')

    cov_right = EyeCov()
    cov_right.scale = 0.5
    cov_right.set_data((arange(N), arange(N)), purpose='sample')
    cov_right.set_data((arange(N), arange(N)), purpose='learn')

    return dict(cov=SumCov([cov_left, cov_right]),
                cov_left=cov_left, cov_right=cov_right)

lik = PoissonProdLik(LogLink())
N = 1000
mean = create_mean(N)
cov = create_cov(N)

y = GGPSampler(lik, mean, cov['cov']).sample(RandomState(0))

ggp = ExpFamGP(y, 'poisson', mean, cov['cov'])

print('Before: %.4f' % ggp.feed().value())
print(cov['cov_left'].scale, cov['cov_right'].scale)

ggp.feed().maximize(progress=True)

print('Before: %.4f' % ggp.feed().value())
print(mean.offset, cov['cov_left'].scale, cov['cov_right'].scale)
