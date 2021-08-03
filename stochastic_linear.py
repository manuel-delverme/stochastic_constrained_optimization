import collections

import experiment_buddy
import torch
import torch.optim
import torch_constrained
import tqdm

n = d = 100
eta = 0.2
loss_scale = 0.05
ITERS = 5000


def stochastic_linear_constraint(tb):
    x = torch.ones((d, 1), requires_grad=True)

    A = torch.zeros(n, d, d)
    for idx in range(n):
        A[idx, idx, idx] = 1.

    def closure_(batch_index):
        Ai = A[batch_index].squeeze(0)

        loss_i = Ai.matmul(x)

        obj_i = loss_scale * loss_i.pow(2).sum()
        constr_i = x.T.matmul(Ai)
        print(obj_i.item(), constr_i.abs().sum().item())

        return obj_i, [constr_i, ], None

    optimizer = torch_constrained.ConstrainedOptimizer(
        torch.optim.SGD,
        torch.optim.SGD,
        lr_x=eta,
        lr_y=eta,
        primal_parameters=[x, ],
    )
    gradnorm = collections.deque(maxlen=d)

    for step in tqdm.trange(ITERS):
        batch_index = torch.randint(0, A.shape[0], (1,))
        closure = lambda: closure_(batch_index)

        lagr = optimizer.step(closure)
        obj, (defect,), _ = closure()

        tb.add_scalar(f"train/mean_defect", float(defect.mean()), step)
        tb.add_scalar(f"train/mean_abs_defect", float(defect.abs().mean()), step)
        tb.add_scalar(f"train/mean_lambda", float(optimizer.equality_multipliers[0].weight.mean()), step)
        tb.add_scalar(f"train/mean_abs_lambda", float(optimizer.equality_multipliers[0].weight.abs().mean()), step)
        tb.add_histogram("train/indices", batch_index, step)

        gradnorm.append(x.grad.abs().sum() + optimizer.equality_multipliers[0].weight.grad.abs().sum())

        if sum(gradnorm) < 1e-10:
            break

    # assert torch.allclose(x, torch.zeros_like(x))


if __name__ == '__main__':
    experiment_buddy.register(locals())
    tb = experiment_buddy.deploy()
    stochastic_linear_constraint(tb)
