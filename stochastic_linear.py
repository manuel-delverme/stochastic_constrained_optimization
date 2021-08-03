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

        dual = optimizer.equality_multipliers[0].weight

        tb.add_scalar(f"primal/obj", float(obj), step)

        tb.add_histogram(f"primal/x", x.detach().numpy(), step)
        tb.add_scalar(f"primal/mean_x", float(x.mean()), step)
        tb.add_scalar(f"primal/mean_abs_x", float(x.mean()), step)

        tb.add_scalar(f"dual/mean_defect", float(defect.mean()), step)
        tb.add_scalar(f"dual/mean_abs_defect", float(defect.abs().mean()), step)

        tb.add_histogram(f"dual/lambda", dual.detach().numpy(), step)
        tb.add_scalar(f"dual/mean_lambda", float(dual.mean()), step)
        tb.add_scalar(f"dual/mean_abs_lambda", float(dual.abs().mean()), step)

        tb.add_scalar(f"train/lagr", float(lagr), step)
        tb.add_histogram("train/indices", batch_index, step)

        for idx in range(10):
            tb.add_scalar(f"{idx}/primal", float(x[idx]), step)
            tb.add_scalar(f"{idx}/dual", float(dual[:, idx]), step)

        gradnorm.append(x.grad.abs().sum() + optimizer.equality_multipliers[0].weight.grad.abs().sum())

        if sum(gradnorm) < 1e-10:
            break

    # assert torch.allclose(x, torch.zeros_like(x))


if __name__ == '__main__':
    experiment_buddy.register(locals())
    tb = experiment_buddy.deploy()
    stochastic_linear_constraint(tb)
