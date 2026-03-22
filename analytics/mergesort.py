import pandas as pd


def _is_nan_like(val) -> bool:
    """Check if a value is NaN or NaT — we need this because NaN doesn't
    compare normally, and we want NaN values to sort to the end (matching
    what pandas does by default).
    """
    try:
        return pd.isna(val)
    except (TypeError, ValueError):
        return False


def _merge(left, right, key, reverse):
    """The merge step of mergesort — combine two already-sorted lists into one.

    NaN values always go to the end regardless of sort direction, which
    keeps us consistent with how pandas handles missing data.
    """
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        lk = key(left[i]) if key else left[i]
        rk = key(right[j]) if key else right[j]

        l_nan = _is_nan_like(lk)
        r_nan = _is_nan_like(rk)

        # NaN values always sink to the bottom
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
    """Sort a list using the classic divide-and-conquer mergesort algorithm.

    Works like Python's built-in sorted() — supports a key function and
    reverse flag. Naturally stable, so equal elements keep their original order.
    """
    lst = list(items)
    if len(lst) <= 1:
        return lst
    mid = len(lst) // 2
    left = mergesort(lst[:mid], key=key, reverse=reverse)
    right = mergesort(lst[mid:], key=key, reverse=reverse)
    return _merge(left, right, key, reverse)


def mergesort_dataframe(df, by, ascending=True):
    """Sort a pandas DataFrame by a column using our custom mergesort.

    This is a drop-in replacement for df.sort_values() — it returns a new
    DataFrame with a clean integer index, same as sort_values + reset_index.
    """
    values = df[by].tolist()
    indexed = list(range(len(values)))

    def key_fn(idx):
        return values[idx]

    sorted_indices = mergesort(indexed, key=key_fn, reverse=not ascending)
    return df.iloc[sorted_indices].reset_index(drop=True)
