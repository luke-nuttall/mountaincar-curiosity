#!/usr/bin/python3

from matplotlib import pyplot as plt
import numpy as np


def plot_gradient_descent(stepsize: float, nsteps: int, filename=None):
    def func(x, y):
        return x**2 + y**2 + (x+y)**2

    def func_deriv(x, y):
        return (
            2*x + 2*(x+y),
            2*y + 2*(x+y),
        )

    def step(x, y):
        grad_x, grad_y = func_deriv(x, y)
        grad_x += (np.random.random()-0.5)*20
        grad_y += (np.random.random()-0.5)*20
        mag = np.sqrt(grad_x**2 + grad_y**2)
        return (
            x - stepsize * grad_x,
            y - stepsize * grad_y,
        )

    xs = np.arange(-5, 5, 0.02)
    ys = np.arange(-5, 5, 0.02)
    grid_xs, grid_ys = np.meshgrid(xs, ys)
    grid_zs = func(grid_xs, grid_ys)

    points = [(-5.1, -3)]
    for ii in range(nsteps):
        x, y = points[-1]
        points.append(step(x, y))

    fig, ax = plt.subplots()
    ax.contour(grid_xs, grid_ys, grid_zs, levels=20)
    ax.plot([p[0] for p in points], [p[1] for p in points], 'o-')
    ax.set_xlim((-5, 5))
    ax.set_ylim((-5, 5))

    plt.tight_layout()

    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)


def main():
    plot_gradient_descent(0.15, 20, "plots/gradient_descent_bigsteps.png")
    plot_gradient_descent(0.01, 50, "plots/gradient_descent_smallsteps.png")


if __name__ == "__main__":
    main()
