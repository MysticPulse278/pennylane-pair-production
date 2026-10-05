import pennylane as qp
import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)

# physical constants
m, e = 1.0, 1.0 
A0, omega, delta = 0.5, 0.5, jnp.pi / 8

# relativistic energy-momentum relation
def En(p): 
    return jnp.sqrt(p[0]**2 + p[1]**2 + m**2)

# pulse envelope
def F(t):
    return 1.0

# 2D Vector potential (elliptically polarized)
def A(t):
    Ax = F(t) * A0 * jnp.cos(omega * t) * jnp.cos(delta)
    Ay = F(t) * A0 * jnp.sin(omega * t) * jnp.sin(delta)
    return jnp.array([Ax, Ay])



# matrix elements
def kappa(p, t):
    E = En(p)
    return 1j * e / E * (jnp.dot(p, A(t)))

def nu(p, t):
    # short notation
    E = En(p)
    A_plus = A(t)[0] + 1j * A(t)[1]
    A_minus = A(t)[0] - 1j * A(t)[1]
    p_plus = p[0] + 1j * p[1]
    p_minus = p[0] - 1j * p[1]
    return 1j * e * (E + m) / (2 * E) * (A_minus - (p_minus / (E + m))**2 * A_plus) * jnp.exp(2j * E * t)


# coefficients for pauli-basis
fx = lambda p, t: -jnp.imag(nu(p, t))
fy = lambda p, t: -jnp.real(nu(p, t))
fz = lambda p, t: -jnp.imag(kappa(p, t))

# time-dependent Hamiltonian 
H = fx * qp.X(0) + fy * qp.Y(0) + fz * qp.Z(0)

dev = qp.device("default.qubit", wires=1)

@jax.jit
@qp.qnode(dev, interface="jax")
def electron_probability(p):
    qp.X(0) # initial condition: |1> (negative energy vacuum state)
    qp.evolve(H)(params=[p, p, p], t=[0, 100])
    return qp.probs(wires=0)

p_vec = jnp.array([2.0, 3.0]) # momentum coordinate

probs = electron_probability(p_vec)
print("Pair creation probability:", probs[0])
