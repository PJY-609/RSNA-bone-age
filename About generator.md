## About generator

* *Example*

```python
from itertools import cycle

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

A = [0, 1, 2, 3, 4, 5, 6, 7]
B = cycle(batch(A, 2))

while True:
    print(next(B))
```

