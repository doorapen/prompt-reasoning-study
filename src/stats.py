def bootstrap_ci(series, n=1000, alpha=0.05):
    """Basic percentile bootstrap for accuracy column of a DataFrame."""
    import numpy as np
    acc = series.values.astype(int)
    boot = [acc[np.random.randint(0, len(acc), len(acc))].mean() for _ in range(n)]
    lower = np.percentile(boot, 100 * alpha / 2)
    upper = np.percentile(boot, 100 * (1 - alpha / 2))
    return lower, upper


def mcnemar(df1, df2):
    """
    Paired McNemar between two strategies on same example set.
    dfX must have columns id, correct (bool).
    """
    import scipy.stats as ss
    merged = df1[["id", "correct"]].merge(df2[["id", "correct"]], on="id", suffixes=("_1", "_2"))
    b = sum((merged.correct_1) & (~merged.correct_2))
    c = sum((~merged.correct_1) & (merged.correct_2))
    stat = (abs(b - c) - 1) ** 2 / (b + c + 1e-9)
    p = ss.chi2.sf(stat, 1)
    return {"b": b, "c": c, "chi2": stat, "p": p} 