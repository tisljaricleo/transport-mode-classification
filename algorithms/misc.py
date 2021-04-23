import numpy as np
import math
from robustats import medcouple


def box_plot(numbers):
    numbers = sorted(numbers)
    q1 = np.percentile(numbers, 25)
    q3 = np.percentile(numbers, 75)
    iqr = q3 - q1
    outliers = [x for x in numbers if not (q1 - 1.5 * iqr) < x < (q3 + 1.5 * iqr)]
    if len(outliers) > 0:
        return outliers
    return None


def sigma(numbers, t):
    numbers = sorted(numbers)
    avg = np.mean(numbers)
    std = np.std(numbers)
    outliers = [x for x in numbers if abs(x - avg) > t * std]
    if len(outliers) > 0:
        return outliers
    return None


def mad(numbers, t):
    numbers = sorted(numbers)
    median = np.median(numbers)
    diff = [abs(x - median) for x in numbers]
    mad_ = np.median(diff)
    coef = t * mad_ / 0.6745
    outliers = [numbers[i] for i in range(len(numbers)) if diff[i] > coef]
    if len(outliers) > 0:
        return outliers
    return None


def adjusted_box_plot(numbers):
    numbers = sorted(numbers)
    q1 = np.percentile(numbers, 25)
    q3 = np.percentile(numbers, 75)
    iqr = q3 - q1
    mc = float(medcouple(numbers))

    if mc >= 0:
        lower = q1 - 1.5 * math.exp(-4 * mc) * iqr
        upper = q3 + 1.5 * math.exp(3 * mc) * iqr
        outliers = [x for x in numbers if not (lower < x < upper)]
    else:
        lower = q1 - 1.5 * math.exp(-3 * mc) * iqr
        upper = q3 + 1.5 * math.exp(4 * mc) * iqr
        outliers = [x for x in numbers if not (lower < x < upper)]

    if len(outliers) > 0:
        return outliers
    return None
