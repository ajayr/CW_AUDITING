class HashTable:
    """A simple hash table built from scratch with open addressing.

    Uses linear probing to handle collisions -- when two keys hash to the
    same slot, we just scan forward until we find an empty one. Automatically
    resizes when the table gets more than 70% full to keep lookups fast.
    """

    _EMPTY = object()
    _DELETED = object()

    def __init__(self, items=None, capacity=16):
        """Set up the table, optionally pre-loading it from a dictionary."""
        self._capacity = max(capacity, 4)
        self._keys = [self._EMPTY] * self._capacity
        self._values = [None] * self._capacity
        self._size = 0
        if items:
            for key, value in items.items():
                self.put(key, value)

    def _hash(self, key):
        """Turn a key into a slot index using Python's built-in hash."""
        return hash(key) % self._capacity

    def _probe(self, key):
        """Walk forward from the hashed slot until we find our key or an empty slot.

        Returns (slot_index, was_found). If we pass a deleted slot on the way,
        we remember it so inserts can reuse that space.
        """
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
        """Double the table size and re-insert everything.

        This gets triggered when load factor exceeds 70% -- without it,
        probe chains get long and lookups slow down.
        """
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
        """Store a key-value pair, overwriting if the key already exists."""
        if self._size / self._capacity > 0.7:
            self._resize()
        idx, found = self._probe(key)
        if not found:
            self._size += 1
        self._keys[idx] = key
        self._values[idx] = value

    def get(self, key, default=None):
        """Look up a key, returning a default value if it's not in the table."""
        idx, found = self._probe(key)
        if found:
            return self._values[idx]
        return default

    def __contains__(self, key):
        """Support 'key in table' checks."""
        _, found = self._probe(key)
        return found

    def __getitem__(self, key):
        """Support table[key] access -- raises KeyError if the key isn't found."""
        idx, found = self._probe(key)
        if not found:
            raise KeyError(key)
        return self._values[idx]

    def __len__(self):
        """How many key-value pairs are currently stored."""
        return self._size
