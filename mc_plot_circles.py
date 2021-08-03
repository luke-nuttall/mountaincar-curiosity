#!/usr/bin/python3

import gym
from matplotlib import pyplot as plt
import numpy as np


def get_track(env, action):
    obs = env.reset()
    track = [obs]
    done = False
    while not done:
        next_obs, reward, done, info = env.step(action)
        track.append(next_obs)
    return track


def main():
    env = gym.make('MountainCar-v0')
    iterations = 200

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax: plt.Axes
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    for action, color in zip((0, 2, 1), ("red", "blue", "green")):
        for ii in range(iterations):
            track = get_track(env, action)
            xs = [obs[0] for obs in track]
            ys = [obs[1] for obs in track]
            ax.plot(xs, ys, color=color, alpha=0.05, zorder=0)
    ax.axhline(y=0, color="black", alpha=0.2)

    ax2 = ax.twinx()
    ax2: plt.Axes
    ax2.set_axis_off()
    xmin = -1.2
    xmax = 0.6
    xs = np.linspace(xmin, xmax, 100)
    ys = np.sin(3 * xs) * .45 + .55
    ax2.plot(xs, ys, "--", color="black")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.04, 0.025)

    plt.tight_layout()
    fig.savefig("plots/circles.png", dpi=300)
    fig.savefig("plots/circles.svg")

    plt.show()


if __name__ == "__main__":
    main()
