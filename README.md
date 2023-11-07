# AlphaZero discrete optimization

## Problem scope

Apply the AlphaZero algorithm to optimization problems of the following form:

- $\mathcal{S}$: a set of states
- $c: \mathcal{S}\to\mathbb{R}$, a cost function
- $\mathcal{A} = \{a_1, \dots, a_A\}$, a finite set of possible actions
  - $\mathcal{A}(s)\subseteq \mathcal{A}$, with abuse of notation, the valid actions from $s$

**Note**: $s\in\mathcal{S}$ is *terminal* if $\mathcal{A}(s)=\emptyset$. For problems without terminal states, we suggest adding a time parameter to the state space, e.g., by letting $\mathbb{T}(\mathcal{S}) := \mathcal{S}\times \mathbb{N}$ and decrementing time accordingly.

## Insights

Consider the action $a$ viable from $s$ which produces $s'$.
How easily can we calculate $c(s')$ from $c(s)$?
It is useful to define the *reward* to be $r(s, a) := c(s) - c(s')$ in this case.

We consider $4$ cases:

1. **visible reward**: $r(s, a)$ can be calculated quickly from $s$ without high computational cost (e.g., allocations, mutations, etc)
2. **side-effect reward**: in the computation which replace $s$ with $s'$, the value $r(s, a)$ is cheap to compute
3. **slow cost**: neither $r(s, a)$ nor $c(s')$ is fast to compute, but $c(s')$ can be computed exactly regardless of $s'$
4. **terminal cost**: when $s'$ is terminal, it is reasonable to compute $c(s')$

todo!()

## License

License

Dual-licensed to be compatible with the Rust project.

Licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) or the [MIT license](https://opensource.org/licenses/MIT), at your option. This file may not be copied, modified, or distributed except according to those terms.
