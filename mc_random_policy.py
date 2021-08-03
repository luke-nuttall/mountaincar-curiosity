#!/usr/bin/python3

import gym
import numpy as np


def play_random(env):
    env.reset()
    done = False
    reward_total = 0
    while not done:
        action = np.random.randint(3)
        next_obs, reward, done, info = env.step(action)
        reward_total += reward
    return reward_total


def main():
    env = gym.make('MountainCar-v0')

    i = 0
    hits = 0
    while True:
        reward = play_random(env)
        if reward > -200:
            hits += 1
        i += 1
        if i % 100 == 0:
            print(f"\rEpisode: {i}, Successes: {hits}", end="")
            # Episode: 2226030, Successes: 0


if __name__ == "__main__":
    main()
