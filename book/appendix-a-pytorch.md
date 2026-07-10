# From BabyTorch to PyTorch

BabyTorch mirrors the *shape* of PyTorch's API, but it is intentionally not a
drop-in replacement. This appendix marks the boundary so that the transition
to production code is predictable.

## The direct translations

| Concept | BabyTorch | PyTorch |
|---|---|---|
| Tensor factory | `babytorch.randn(2, 3)` | `torch.randn(2, 3)` |
| Trainable layer | `nn.Linear(3, 8)` | `torch.nn.Linear(3, 8)` |
| Forward call | `y = model(x)` | `y = model(x)` |
| Clear gradients | `optimizer.zero_grad()` | `optimizer.zero_grad()` |
| Reverse pass | `loss.backward()` | `loss.backward()` |
| Parameter update | `optimizer.step()` | `optimizer.step()` |
| Evaluation mode | `model.eval()` | `model.eval()` |
| Disable graph recording | `with babytorch.no_grad():` | `with torch.no_grad():` |

The standard training loop therefore moves almost unchanged:

```python
for x, y in loader:
    prediction = model(x)
    loss = criterion(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## The differences that matter

1. **Devices belong to tensors in PyTorch.** BabyTorch selects one global
   NumPy, CuPy, or experimental MLX backend before tensors are created.
   PyTorch moves individual tensors and modules with `.to(device)`.
2. **Dtypes are richer in PyTorch.** BabyTorch is designed around floating
   arrays and uses float32 factories. PyTorch preserves integer tensors,
   supports several reduced-precision formats, and enforces dtype/device rules.
3. **Autograd is more strict and more configurable in PyTorch.** BabyTorch
   seeds a missing output gradient with ones for teaching convenience. PyTorch
   requires an explicit gradient for non-scalar outputs and normally frees a
   graph after use unless `retain_graph=True` is requested.
4. **Modules register state explicitly in PyTorch.** BabyTorch discovers
   trainable tensors by recursively walking attributes. PyTorch distinguishes
   parameters, buffers, and submodules and serializes a named `state_dict`.
5. **Performance is not comparable.** BabyTorch exposes operations so they can
   be studied. PyTorch fuses, compiles, dispatches, and parallelizes them.

## A safe migration checklist

- Replace `import babytorch` with `import torch` and use `torch.nn` and
  `torch.optim`.
- Choose a device, then move both model and batches to it.
- Check tensor dtypes, especially class labels (`torch.long`) and mixed
  precision.
- Replace raw `.data` mutation with `torch.no_grad()` or optimizer operations.
- Save and load `state_dict()` objects instead of pickled model internals.
- Re-run shape, gradient, and end-to-end learning tests before scaling up.

The important transfer is conceptual: when PyTorch presents a larger surface,
you now know which part records a graph, which part owns parameters, which part
computes local derivatives, and which part updates weights.
