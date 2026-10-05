# Quantum simulation of pair creation in polarized electric fields

The momentum-dependent electron-positron pair creation in time-dependent elliptically polarized electric field pulses is simulated using **PennyLane** and **JAX**.

The time evolution of the relativistic Dirac equation is mapped to an $SU(2)$ spin system and simulated on a single-qubit quantum circuit.

The elliptically polarized electric field is defined in the $xy$-plane by vector potential $\mathbf{A}(t) = (A_x(t), A_y(t))$. It leads to a 2-level quantum state (vacuum state $|1>$ and pair-created state $|0>$) dependent on the momentum $\mathbf{p}$ of the created electron. 

The transition between the vacuum and pair creation state is governed by two matrix elements of the Hamiltonian, $\kappa$ and $\nu$. Both were derived for arbitrarily polarized vector potentials in the $xy$-plane.

By expanding the Hamiltonian in the Pauli basis $\{\sigma_x, \sigma_y, \sigma_z\}$ a simulation on a single-qubit quantum circuit using Pauli gates $X$, $Y$ and $Z$ becomes possible. The circuit is simulated using GPU acceleration via $JAX$.

As a carrier, a 2D harmonic oscillator with amplitude $A_0$ and frequency $\omega$ is chosen. The parameter $\delta$ controls the polarization type. $\delta = 0$ corresponds to linear polarization along $x$-axis, whereas $\delta = \pi / 4$ corresponds to circular polarization.

# TODO

- check for unitarity (probabilities of both states should add up to 1)
- define a finite field pulse with start and end time
- plot 2D momentum distribution spectra in the $p_xp_y$-plane


