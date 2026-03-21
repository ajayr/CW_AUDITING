class HashTable:
    """Hash table with open addressing (linear probing) for collision resolution."""

    _EMPTY = object()
    _DELETED = object()

    def __init__(self, items=None, capacity=16):
        self._capacity = max(capacity, 4)
        self._keys = [self._EMPTY] * self._capacity
        self._values = [None] * self._capacity
        self._size = 0
        if items:
            for key, value in items.items():
                self.put(key, value)

    def _hash(self, key):
        return hash(key) % self._capacity

    def _probe(self, key):
        """Linear probing: find the slot for key, handling collisions."""
        idx = self._hash(key)
        first_deleted = None
        for _ in range(self._capacity):
            if self._keys[idx] is self._EMPTY:
                return (first_deleted if first_deleted is not None else idx), False
            if self._keys[idx] is self._DELETED:
                if first_deleted is None:
                    first_deleted = idx
            elif self._keys[idx] == key:
                return idx, True
            idx = (idx + 1) % self._capacity
        return (first_deleted if first_deleted is not None else -1), False

    def _resize(self):
        old_keys = self._keys
        old_values = self._values
        self._capacity *= 2
        self._keys = [self._EMPTY] * self._capacity
        self._values = [None] * self._capacity
        self._size = 0
        for k, v in zip(old_keys, old_values):
            if k is not self._EMPTY and k is not self._DELETED:
                self.put(k, v)

    def put(self, key, value):
        """Insert or update a key-value pair."""
        if self._size / self._capacity > 0.7:
            self._resize()
        idx, found = self._probe(key)
        if not found:
            self._size += 1
        self._keys[idx] = key
        self._values[idx] = value

    def get(self, key, default=None):
        """Retrieve value by key, returning default if not found."""
        idx, found = self._probe(key)
        if found:
            return self._values[idx]
        return default

    def __contains__(self, key):
        _, found = self._probe(key)
        return found

    def __getitem__(self, key):
        idx, found = self._probe(key)
        if not found:
            raise KeyError(key)
        return self._values[idx]

    def __len__(self):
        return self._size
