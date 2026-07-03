from __future__ import annotations

try:
    from tqdm import tqdm as tqdm
except ImportError:

    class _NoOpProgress:
        def __init__(self, *args, **kwargs):
            self.iterable = args[0] if args else None

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)

        def update(self, n: int = 1) -> None:
            return None

        def close(self) -> None:
            return None

    def tqdm(*args, **kwargs):
        return _NoOpProgress(*args, **kwargs)
