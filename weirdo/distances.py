def hamming(p1, p2):
    n = min(len(p1), len(p2))
    return sum([p1[i] != p2[i] for i in range(n)])

