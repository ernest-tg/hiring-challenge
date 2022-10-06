import torch


def compute_average_precision(precisions, recalls):
    previous_r = recalls[0]
    area_under_curve = 0
    for p, r in zip(precisions[1:], recalls[1:]):
        if torch.isnan(p) or torch.isnan(r):
            return torch.nan
        assert 0 <= p <= 1, f"{p} is not a valid precision"
        assert 0 <= r <= 1, f"{r} is not a valid recall"
        assert previous_r >= r, f"Recalls should be sorted in descending order, {previous_r} > {r}"
        area_under_curve += p * (previous_r-r)
        previous_r = r
    return area_under_curve
