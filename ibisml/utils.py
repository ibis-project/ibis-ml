def encode_labels(table, columns=None, labels=None):
    if columns is not None:
        if labels is not None:
            raise ValueError("Cannot pass both `columns` and `labels`")
        labels = {}
        if isinstance(columns, str):
            columns = [columns]
        for c in columns:
            labels[c] = list(table.select(c).distinct().order_by(c).execute()[c])
    elif labels is None:
        raise ValueError("Must pass either `columns` or `labels`")

    encoded = table.mutate(
        [table[c].find_in_set(fs).name(c) for c, fs in labels.items()]
    )
    return encoded, labels
