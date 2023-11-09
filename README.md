# AlphaZero discrete optimization

![a 3d plot showing 64 independent searches with `(x, y, z)` axes labeled `(Episodes, Matching, Lambda)`](readme_3d_cost.png)

The goal of this project is to optimize arbitrary functions over high-dimensional discrete state spaces with limited prior knowledge through self-play. See the [Issues](https://github.com/ariasanovsky/azdopt/issues/) for planned features and optimizations. Hyperparameters are not optimized.

Try the [examples](https://github.com/ariasanovsky/azdopt/blob/main/graph-state/examples/06-c21.rs) yourself.

It combines ideas from the papers:

- [Constructions in combinatorics via neural networks](https://arxiv.org/abs/2104.14516), and
- [Alpha Zero](https://arxiv.org/pdf/1712.01815.pdf).

It mainly uses:

- [`dfdx`](https://docs.rs/dfdx/latest/dfdx/) for type-checked neural networks,
- [`rayon`](https://docs.rs/rayon/latest/rayon/) for data-parallel MCTS exploration,
- and `python` for data visualization.

## Contributions

Feedback on a high-level API is welcome. Please make Github Issues before making PRs.

## Alpha Zero algorithm

Our first implementation can be summarized:

1. Our agents search trees in parallel.
2. During episodes, they seek to maximize the expected future improvement in cost function *over the entire search path*.
   1. They end each episodes when they visit new nodes.
   2. The neural network evaluates state vectors in batches.
   3. Each evaluation predicts the gain function and the agent's preference for actions from the new state.
3. Between episodes, we populate new nodes with their predictions and update nodes along the path.
4. Between epochs, we minimize cross entropy loss (priors' inaccuracy) and $L_2$ loss (gain prediction inaccuracy).
   1. Additionally, we select new roots from visited nodes.
   2. When stagnant, we generate new roots.

## States and Actions

We restrict our attention to state spaces with a finite number of dimensions whose actions are discrete and commonly enumerated. Our most thorough examples uses the `INTMinTree`, as demonstrated [here](https://github.com/ariasanovsky/azdopt/blob/main/graph-state/examples/06-c21.rs). Examples `01` through `04` have prototypes to re-implement similarly.

<!-- Apply the AlphaZero algorithm to optimization problems of the following form:

- $\mathcal{S}$: a set of states
- $c: \mathcal{S}\to\mathbb{R}$, a cost function
- $\mathcal{A} = \{a_1, \dots, a_A\}$, a finite set of possible actions
  - $\mathcal{A}(s)\subseteq \mathcal{A}$, with abuse of notation, the valid actions from $s$

**Note**: $s\in\mathcal{S}$ is *terminal* if $\mathcal{A}(s)=\emptyset$. For problems without terminal states, we suggest adding a time parameter to the state space, e.g., by letting $\mathbb{T}(\mathcal{S}) := \mathcal{S}\times \mathbb{N}$ and decrementing time accordingly.

## Insights

Consider the action $a$ viable from $s$ which produces $s'$.
How easily can we calculate $c(s')$ from $c(s)$?
It is useful to define the *reward* to be $r(s, a) := c(s) - c(s')$ in this case. -->

We consider a few variants based on how the cost function is calculated:

1. **visible reward**: $r(s, a)$ can be calculated quickly from $s$ without high computational cost (e.g., allocations, mutations, etc)
2. **side-effect reward**: in the computation which replace $s$ with $s'$, the value $r(s, a)$ is cheap to compute, but vector $r(s, \cdot)$ is not cheap.
   1. **slow cost** (similar to the above): neither $r(s, a)$ nor $c(s')$ is fast to compute, but $c(s')$ can be computed exactly regardless of $s'$.
3. **terminal cost**: only when $s'$ is terminal, it is reasonable to compute $c(s')$

## Termination

For states which have no terminal nodes, we impose timers and grow a list of *prohibited* actions. For example, when modify a graph, we are not allowed to modify the same edge twice.

## License

License

Dual-licensed to be compatible with the Rust project.

Licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) or the [MIT license](https://opensource.org/licenses/MIT), at your option. This file may not be copied, modified, or distributed except according to those terms.
