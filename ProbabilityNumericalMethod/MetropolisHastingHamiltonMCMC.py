import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

#1: METROPOLIS algorithm
#p(x) = (1+ x^2)^(-1)
# To do so using metropolis sampling, we need two things: (1) the prior distribution over initial states
#                                                         (2) the proposal distribution q(x|x(t-1)) - symmetry
#prior ~ N(0,1)
#q(x|x(t-1)) ~ N(x(t-1), 1)

# np.random.seed(1234)
#
# def target_distribution(x):
#     return (1 + x**2)**(-1)
#
# n_samples = 10000
# nDisplay  = 20
# sigma     = 1
# minn      = -20
# maxx      = 20
# pauseDur = 1.0
# x_grid = np.linspace(3*minn, 3*maxx, 1200)
# target = target_distribution(x_grid)
#
# #initialize sampler
# x = np.zeros((1, n_samples))
# x[:, 0] = np.random.randn()
#
# #accepted count
# accepted_count = 0
# x_accepted = []
#
#
# for t in range(n_samples-1):
#
#     #sample from random proposal
#     xStar = np.random.normal(x[:, t], sigma)
#     proposal = norm.pdf(x_grid, x[:, t], sigma)
#
#     #calculate the acceptance probability
#     alpha = min(1.0, target_distribution(xStar)/target_distribution(x[:, t]))
#
#     #accept and reject
#     u = np.random.uniform()
#     if u < alpha:
#         x[:, t+1] = xStar
#         str = 'Accepted'
#         accepted_count += 1
#         x_accepted.append(xStar)
#     else:
#         x[:, t+1] = x[:, t]
#         str = 'Rejected'
#
#     if t < (nDisplay + 1):
#
#         plt.cla()
#         plt.xlim(minn, maxx)
#         plt.plot(x_grid, target, 'g')
#         plt.hold(True)
#         plt.plot(x_grid, proposal, 'r')
#         plt.hold(True)
#         plt.plot(x_grid, mlab.normpdf(x_grid, x[:, t], sigma), 'b')
#         plt.hold(True)
#         plt.plot([x[:, t], x[:, t]], [0, target_distribution(x[:, t])], 'b')
#         plt.hold(True)
#         plt.plot([xStar, xStar], [0, target_distribution(xStar)], 'r')
#         plt.hold(True)
#         plt.scatter(xStar, 0, s = 20, c = 'r')
#         plt.hold(True)
#         zero_y = np.zeros((1, x[:, 0:t].shape[1]))
#         plt.scatter (x[:, 0:t], zero_y)
#         plt.pause(pauseDur)
#
# print('ratio between accepted and samples is %f'%(accepted_count/n_samples))
# print(len(x_accepted))
#
# x_draw = np.reshape(x, (n_samples,))
# plt.hist(x_accepted, bins = 100, range = [minn, maxx])
# plt.show()


#Metroplis-Hasting algorithm
#the metropolis-hastings algorithm is implemented with essentially the same procedure as the Metropolis sampler,
# except that the correction factor is used (c = q(x(t-1)|x*)/ q(x*|x(t-1)). Specifically, to draw M samples using the Metropolis-Hasting sampler:
# Step1: set t = 0
# Step2: generate an initial state x(0)
# Step3: Repeat until t = M
#        set t = t + 1
#        generate a proposal state x* from q(x|x(t-1))
#        calculate the proposal correction factor c = q(x(t-1)|x*) / q(x*|x(t-1))
#        calculate the acceptance probability anpha = min (1, p(x*)/p(x(t-1)) x c)
#        draw a random number u from Unif(0,1)
#               if u <= anpha accept the proposal state x* and set x(t) = x*
#               else set x(t) = x(t-1)




#Hamiltonian Monte Carlo
#Using Hamiltonian MC for sampling a Bivariate Normal Distribution
#p(x) = N(u, sigma)
# u = [u1, u2] = [0, 0]
# sigma = [[1, 0.8] [0.8, 1]]

#IMPLEMENTATION
np.random.seed(123)
np.random.seed(61214)

#step size
delta = 0.3
nSamples = 1000
L = 20

def potential_energy(x):
    """
    define potential energy function based on x input vector
    :param x: 2D input vector
    :return:
    """
    sigma = np.array([[1, 0.8],[0.8, 1]]).astype("float32")
    xT_dot_inv_sigma = np.dot(x.transpose(), np.linalg.inv(sigma))
    energy = xT_dot_inv_sigma.dot(x)
    return energy

def gradient_potential_energy(x):
    """

    :param x:
    :return:
    """
    sigma = np.array([[1, 0.8], [0.8, 1]]).astype("float32")
    dU =  x.transpose().dot(sigma)
    dU_return = dU.reshape(-1, 1)
    return dU_return

def kinetic_energy(p):
    return np.sum(p.transpose().dot(p))*0.5

#initial state
x = np.zeros((2, nSamples)).astype("float32")
x0 = np.array([[0, 6]])
x[:, 1] = x0


for t in np.arange(1, nSamples-1):

    p0 = np.random.randn(2,1)

    pStar = p0 - (delta/2)*gradient_potential_energy(x[:, t])

    xStar = x[:, t].reshape(-1, 1) + delta*pStar

    for j in np.arange(1, L):
        pStar = pStar - delta*gradient_potential_energy(xStar)

        xStar = xStar + delta*pStar

    pStar = pStar - (delta/2)*gradient_potential_energy(xStar)

    U0 = potential_energy(x[:, t])
    Ustar = potential_energy(xStar)
    #print(U0)
    #print(Ustar)

    K0 = kinetic_energy(p0)
    KStar = kinetic_energy(pStar)

    alpha = min(1, np.exp((U0 + K0) - (Ustar + KStar)))

    u = np.random.uniform()

    if u < alpha:
        x[:, t + 1] = xStar.reshape(2,)
    else:
        x[:, t + 1] = x[:, t]

plt.xlim(-6.0, 6.0)
plt.ylim(-6.0, 6.0)
plt.plot(x[0, 1:50], x[1, 1:50], 'b')
plt.scatter(x[0, :], x[1, :], s = 20, c = 'r')
plt.show()
