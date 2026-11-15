# golden_section_search_py

Minimizes a unimodal scalar function on a closed interval using only function evaluations, with no derivatives required.

```python
from golden_section_search_py import golden_section_search

result = golden_section_search(lambda x: (x - 3.0) ** 2 + 1.0, -10.0, 10.0)
print(result.x)          # ~3.0
print(result.fx)         # ~1.0
print(result.iterations) # number of shrinking steps
print(result.evaluations)# number of function calls
```

## Why this exists

Derivative-free minimization on an interval is a common need when the objective is a black box, noisy, or expensive to differentiate. The golden section search is the classic answer: it shrinks the bracket by a fixed factor per iteration and reuses one function value, so it needs only one new evaluation per step. The trade-off is that it assumes unimodality. If the function has multiple local minima, the search will converge to whichever basin contains the initial interval's interior points; it does not attempt global optimization.

## API

The package exports three names:

- `golden_section_search(f, a, b, *, tol=1e-8, max_iter=200)` — returns a `GoldenSectionSearchResult`.
- `GoldenSectionSearchResult` — a frozen dataclass with fields `x`, `fx`, `iterations`, and `evaluations`.
- `GoldenSectionSearchError` — a `ValueError` subclass raised for invalid arguments.

`f` must be a callable taking one float and returning a float. `a` must be strictly less than `b`. `tol` must be positive and `max_iter` must be at least 1; otherwise `GoldenSectionSearchError` is raised. The search stops when the interval width is at most `tol` or after `max_iter` iterations, whichever comes first.

## The awkward edge

When the function is monotone on the interval, the minimum is at an endpoint. The algorithm still runs and returns a point very close to that endpoint, not exactly the endpoint, because it only ever evaluates interior points. If you need the exact endpoint, check `f(a)` and `f(b)` yourself and compare. The returned `x` is always an interior point that was actually evaluated, so `fx` is a real function value and not an interpolation.
