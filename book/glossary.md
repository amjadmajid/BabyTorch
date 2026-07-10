# Glossary

**Action** — A choice made by an agent in an environment.

**Activation** — Either the output value of a layer or a non-linear function
such as ReLU or GELU, depending on context.

**Autograd** — A system that records tensor operations and applies the chain
rule in reverse to compute gradients.

**Backward pass** — The reverse traversal that propagates a loss gradient
through a computation graph.

**Batch** — A group of examples processed together in one forward pass.

**Bellman equation** — A recursive relationship between present reward and the
discounted value of a future state or action.

**Broadcasting** — Treating size-one or missing tensor dimensions as repeated
without physically copying their values.

**Causal mask** — A mask that prevents a sequence position from attending to
future positions.

**Computation graph** — The directed record of operations connecting input
tensors to an output.

**Denoiser** — In diffusion, a model that predicts added noise (or an equivalent
clean-data quantity) from a noisy sample and timestep.

**Embedding** — A learned lookup table that replaces an integer id with a
dense vector.

**Epoch** — One pass through a training dataset.

**Gradient** — The derivative of an output, usually the loss, with respect to
one or more values.

**Inference** — Using a trained model without updating its parameters.

**Leaf tensor** — A graph input created directly rather than by another tracked
operation; model parameters are leaf tensors.

**Learning rate** — The scale of an optimizer's parameter update.

**Logit** — An unnormalized score, commonly converted to probabilities by
softmax.

**Loss** — A scalar objective that measures prediction error or another
training goal.

**Parameter** — A tensor learned by an optimizer.

**Policy** — In reinforcement learning, a rule or distribution that selects an
action from a state.

**Return** — The discounted sum of rewards from a point in a trajectory.

**Tensor** — An n-dimensional array together with its shape and, in an autograd
framework, optional gradient history.

**Token** — A discrete unit of text represented by an integer id.

**Transformer** — A neural architecture built from attention, position-wise
MLPs, residual paths, and normalization.

**Validation set** — Data held out from parameter updates and used to estimate
generalization during development.

