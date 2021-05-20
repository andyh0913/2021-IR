def calc_ap(topk, positive):
    ap = 0
    count = 0
    for i, item in enumerate(topk):
        if item in positive:
            count += 1
            ap += count/(i+1)
    ap /= (len(positive) + 1e-8)
    
    return ap