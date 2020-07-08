# adding this to check github integration on slack
# TODO: move general methods into here


# https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b        
def topk_acc(output, target, topk=(1,), device=None):
    """Computes the standard topk accuracy for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    targ = target.unsqueeze(1).repeat(1,maxk).to(device)
    correct = pred.eq(targ)
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0).cpu()
        res.append(correct_k.mul_(100.0 / batch_size))
    del targ, pred, target
    return res
