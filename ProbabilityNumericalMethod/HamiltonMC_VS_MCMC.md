The random-walk behavior of many MCMC algorithms make the markov chain convergence to the target distribution p(x) inefficiently. 
The Hamiltonian MC algorithm, is a type of MCMC method that adopts dynammics physical system to propose future states of the system rather than
proposal probability distribution. This allows Markov chain to explore the target distribution much more efficiently and faster to target distribution.
In this demo, we gonna show how Hamiltonian dynamics can be used as MC proposal function for an MCMC sampling algorithm.

#Hamiltonian Dynamics
H(x,p) = U(x) + K(p)

U(x): potential energy
K(p): kinematic energy

Hamiltonian dynamics describes how kinematic energy converges to potential energy (and vice versa)

 This description is implemented quantitatively via a set of differential equations known as the Hamiltonian equations:

d(x_i)/dt = dH/d(p_i) = dK/d(p_i)
d(p_i)/dt = dH/d(x_i) = dU/d(x_i)

