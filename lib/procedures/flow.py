import torch
import matplotlib.pyplot as plt
from torch import nn
import numpy as np


class Pruner:
    def __init__(self):
        self.final_scores = {}
        self.scores_temp = {}
        self.scores = {}
        self.index = True

    def score(self, model, loss, dataloader, device):
        raise NotImplementedError

    def _global_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level globally.
        """
        # # Set score for masked parameters to -inf
        # for mask, param in self.masked_parameters:
        #     score = self.scores[id(param)]
        #     score[mask == 0.0] = -np.inf

        # Threshold scores
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        k = int((1.0 - sparsity) * global_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            for mask, param in self.masked_parameters:
                score = self.scores[id(param)]
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def _local_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level parameter-wise.
        """
        for mask, param in self.masked_parameters:
            score = self.scores[id(param)]
            k = int((1.0 - sparsity) * score.numel())
            if not k < 1:
                threshold, _ = torch.kthvalue(torch.flatten(score), k)
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def mask(self, sparsity, scope):
        r"""Updates masks of model with scores by sparsity according to scope.
        """
        if scope == 'global':
            self._global_mask(sparsity)
        if scope == 'local':
            self._local_mask(sparsity)

    @torch.no_grad()
    def apply_mask(self):
        r"""Applies mask to prunable parameters.
        """
        for mask, param in self.masked_parameters:
            param.mul_(mask)

    def alpha_mask(self, alpha):
        r"""Set all masks to alpha in model.
        """
        for mask, _ in self.masked_parameters:
            mask.fill_(alpha)

    # Based on https://github.com/facebookresearch/open_lth/blob/master/utils/tensor_utils.py#L43
    def shuffle(self):
        for mask, param in self.masked_parameters:
            shape = mask.shape
            perm = torch.randperm(mask.nelement())
            mask = mask.reshape(-1)[perm].reshape(shape)

    def invert(self):
        for v in self.scores.values():
            v.div_(v ** 2)

    def stats(self):
        r"""Returns remaining and total number of prunable parameters.
        """
        remaining_params, total_params = 0, 0
        for mask, _ in self.masked_parameters:
            remaining_params += mask.detach().cpu().numpy().sum()
            total_params += mask.numel()
        return remaining_params, total_params

def draw_scatter(x, y):
    index = 0
    x_ = []
    y_ = []
    for ind in range(len(x)):
        x_.append(x[ind])
        for i in range(len(x[ind])):
            y_.append(index)
        index += 1
    # length = len(x)
    # y = np.zeros(length)
    # plt.xlim((-1, 1))
    plt.scatter(x_, y_)
    plt.show()

def count(labels, item):
    num = 0
    index = []
    for i, lable in enumerate(labels):
        if item == lable:
            num += 1
            index.append(i)
    return num, index

def score(models, loss, input_ori, target, dataloader,  device):

    score = []
    for model in models:
        model.train()

    # compute gradient
    # for batch_idx, (data, target) in enumerate(dataloader):
    #     data, target = data.to(device), target.to(device)
        # print(data.shape)
    res_var = []
    for model in models:
        input_ori, target = input_ori.to(device), target.to(device)
        jacobs = []
        targets = []

        model.zero_grad()
        input_ori.requires_grad_(True)
        _, output = model(input_ori)
        output.backward(torch.ones_like(output))
        jacob = input_ori.grad.detach()
        jacobs.append(jacob.reshape(jacob.size(0), -1).cpu().numpy())
        jacobs = np.concatenate(jacobs, axis=0)
        targets.append(target.cpu().numpy())


        k = 1e-5
        corrs = np.corrcoef(jacobs)
        s = np.sum(np.log(abs(corrs) + k))
        s = s / len(corrs)
        score.append(np.absolute(s))

        # bool = [True] * len(target)
        # jacob_mv = []
        # for i, lable in enumerate(target):
        #     num, index = count(target, lable)
        #     if num > 1:
        #         if bool[i] == True:
        #             jacob_mv.append(jacobs[i])
        #             for inx in index:
        #                 bool[inx] = False
        #     else:
        #         jacob_mv.append(jacobs[i])
        # corrs = np.corrcoef(jacob_mv)
        # s = np.sum(np.log(abs(corrs)+k))
        # s = s / len(corrs)
        # score = np.absolute(s)
    return score, res_var



class Rand(Pruner):
    def __init__(self, masked_parameters):
        super(Rand, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.randn_like(p)


class Mag(Pruner):
    def __init__(self, masked_parameters):
        super(Mag, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.data).detach().abs_()


# Based on https://github.com/mi-lad/snip/blob/master/snip.py#L18
class SNIP(Pruner):
    def __init__(self):
        super(SNIP, self).__init__()

    def score(self, model, loss, dataloader, device):

        # allow masks to have gradient
        losses = []
        idx = 0
        for model_ in model:
            model_.train()
            for m, w in model_.named_parameters():
                w.requires_grad = True

        # compute gradient
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            for model_ in model:
                output = model_(data)
                if isinstance(output, tuple):
                    output = output[1]
                loss = loss(output / 0.05, target)
                losses.append(loss)
            idx += 1
            if idx > 0:
                return losses

        # calculate score |g * theta|
        # for m, p in self.masked_parameters:
        #     self.scores[id(p)] = torch.clone(m.grad).detach().abs_()
        #     p.grad.data.zero_()
        #     m.grad.data.zero_()
        #     m.requires_grad = False
        #
        # # normalize score
        # all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        # norm = torch.sum(all_scores)
        # for _, p in self.masked_parameters:
        #     self.scores[id(p)].div_(norm)


# Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L49
class GraSP(Pruner):
    def __init__(self, masked_parameters):
        super(GraSP, self).__init__(masked_parameters)
        self.temp = 200
        self.eps = 1e-10

    def score(self, model, loss, dataloader, device):

        # first gradient vector without computational graph
        stopped_grads = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data) / self.temp
            L = loss(output, target)

            grads = torch.autograd.grad(L, [p for (_, p) in self.masked_parameters], create_graph=False)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
            stopped_grads += flatten_grads

        # second gradient vector with computational graph
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data) / self.temp
            L = loss(output, target)

            grads = torch.autograd.grad(L, [p for (_, p) in self.masked_parameters], create_graph=True)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])

            gnorm = (stopped_grads * flatten_grads).sum()
            gnorm.backward()

        # calculate score Hg * theta (negate to remove top percent)
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p.data).detach()
            p.grad.data.zero_()

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.abs(torch.sum(all_scores)) + self.eps
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


class SynFlow(Pruner):
    def __init__(self, name_op):
        super(SynFlow, self).__init__(name_op)

    def score(self, model, loss, dataloader, device):
        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():  # conv and linear param
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])  # all abs param = [-1, 0, 1]
                print(param)


        # signs = linearize(model)

        model.train()
        (data, _) = next(iter(dataloader))
        input_dim = list(data[0, :].shape)
        inputs = torch.ones([1] + input_dim).to(device) # , dtype=torch.float64).to(device)
        input = inputs.clone().cuda(device=device, non_blocking=True)
        output = model(input)
        model.zero_grad()
        if isinstance(output, tuple):
            output = output[1]
        torch.sum(output).backward(torch.ones_like(torch.sum(output)), retain_graph=True)
        for name, W in model.named_parameters():
            if 'weight' in name and W.grad is not None:
                self.scores_temp[name] = torch.clone(W.grad * W).detach().abs_()
                W.grad.data.zero_()
        for name, score in self.scores_temp.items():
            self.scores[name] = torch.sum(score) / score.numel()
        for name, score in self.scores.items():
            if name[8:25] in self.final_scores.keys():
                self.final_scores[name[8:25]] += score
            else:
                self.final_scores[name[8:25]] = score
        return self.final_scores




