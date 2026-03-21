import pandas as pd


def _is_nan_like(val) -> bool:
    """Check if a value is NaN or NaT (sorts last, matching pandas default)."""
    try:
        return pd.isna(val)
    except (TypeError, ValueError):
        return False


def _merge(left, right, key, reverse):
    """Merge two sorted lists into one sorted list."""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        lk = key(left[i]) if key else left[i]
        rk = key(right[j]) if key else right[j]

        l_nan = _is_nan_like(lk)
        r_nan = _is_nan_like(rk)

        if l_nan and r_nan:
            pick_left = True
        elif l_nan:
            pick_left = False
        elif r_nan:
            pick_left = True
        elif reverse:
            pick_left = lk >= rk
        else:
            pick_left = lk <= rk

        if pick_left:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result


def mergesort(items, key=None, reverse=False):
    """Divide-and-conquer mergesort. Returns a new sorted list.

    Args:
        items: Iterable to sort.
        key: Optional function to extract comparison key (like sorted()).
        reverse: If True, sort descending.
    """
    lst = list(items)
    if len(lst) <= 1:
        return lst
    mid = len(lst) // 2
    left = mergesort(lst[:mid], key=key, reverse=reverse)
    right = mergesort(lst[mid:], key=key, reverse=reverse)
    return _merge(left, right, key, reverse)


def mergesort_dataframe(df, by, ascending=True):
    """Sort a pandas DataFrame using mergesort on the given column.

    Args:
        df: DataFrame to sort.
        by: Column name to sort by.
        ascending: Sort ascending (True) or descending (False).

    Returns:
        A new DataFrame sorted by the specified column, with reset-ready index.
    """
    values = df[by].tolist()
    indexed = list(range(len(values)))

    def key_fn(idx):
        return values[idx]

    sorted_indices = mergesort(indexed, key=key_fn, reverse=not ascending)
    return df.iloc[sorted_indices].reset_index(drop=True)
