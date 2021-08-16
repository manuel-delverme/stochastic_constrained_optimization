import collections

import experiment_buddy
import torch
import torch.optim
import torch_constrained
import tqdm

n = d = 100
eta = 0.2
loss_scale = 0.
constr_scale = 1.
iterations = 10000


def stochastic_linear_constraint(logger):
    plot_every = iterations // 1000

    x = torch.ones((d, 1), requires_grad=True)

    A = torch.zeros(n, d, d)
    for idx in range(n):
        A[idx, idx, idx] = 1.

    def closure_(batch_index_):
        Ai = A[batch_index_].squeeze(0)

        # loss_i = Ai.matmul(x)
        obj_i = 0.  # loss_scale * loss_i.pow(2).sum()
        constr_i = constr_scale * x.T.matmul(Ai)

        return obj_i, [constr_i, ], None

    optimizer = torch_constrained.ConstrainedOptimizer(
        torch_constrained.ExtraSGD,
        torch_constrained.ExtraSGD,
        lr_x=eta,
        lr_y=eta,
        primal_parameters=[x, ],
    )
    grad_norm = collections.deque(maxlen=d)

    for step in tqdm.trange(iterations):
        batch_index = torch.randint(0, A.shape[0], (1,))
        closure = lambda: closure_(batch_index)

        lagrangian = optimizer.step(closure)  # noqa

        if (step % plot_every) == 0:
            dual = optimizer.equality_multipliers[0].weight
            for idx in range(10):
                logger.add_scalar(f"{idx}/primal", float(x[idx]), step)
                logger.add_scalar(f"{idx}/dual", float(dual[:, idx]), step)

                logger.add_scalar(f"{idx}/grad_primal", float(x.grad[idx]), step)
                logger.add_scalar(f"{idx}/grad_dual", float(dual.grad[:, idx]), step)
                logger.add_scalar(f"{idx}/selected", float(batch_index == idx), step)

            obj, (defect,), _ = closure()

            logger.add_scalar(f"primal/obj", float(obj), step)

            logger.add_histogram(f"primal/x", x.detach().numpy(), step)
            logger.add_scalar(f"primal/mean_x", float(x.mean()), step)
            logger.add_scalar(f"primal/mean_abs_x", float(x.mean()), step)

            logger.add_scalar(f"dual/mean_defect", float(defect.mean()), step)
            logger.add_scalar(f"dual/mean_abs_defect", float(defect.abs().mean()), step)

            logger.add_histogram(f"dual/lambda", dual.detach().numpy(), step)
            logger.add_scalar(f"dual/mean_lambda", float(dual.mean()), step)
            logger.add_scalar(f"dual/mean_abs_lambda", float(dual.abs().mean()), step)

            logger.add_scalar(f"train/lagrangian", float(lagrangian), step)
            logger.add_histogram("train/indices", batch_index, step)

        grad_norm.append(x.grad.abs().sum() + optimizer.equality_multipliers[0].weight.grad.abs().sum())

        if sum(grad_norm) < 1e-10:
            break


if __name__ == '__main__':
    experiment_buddy.register(locals())
    tb = experiment_buddy.deploy()
    stochastic_linear_constraint(tb)
