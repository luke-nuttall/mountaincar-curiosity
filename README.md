# Solving MountainCar with Intrinsic Curiosity
This is a TensorFlow implementation of an agent which can learn to solve the MountainCar reinforcement learning problem from the gym library.

MountainCar is a deceptively tricky RL problem with sparse rewards, meaning it can't be solved in a reasonable amount of time via a naive Q-learning approach.

This implementation is based on the Intrinsic Curiosity Module (ICM) developed by Pathak et al. https://pathak22.github.io/noreward-rl/. The ICM provides an intrinsic reward which is added to the extrinsic reward as input to a conventional DQN architecture. This allows the Q network to learn exploration behavior and discover the goal far faster than would be possible with random exploration.

A full technical writeup of this project can be found at https://hackmd.io/@luke-nuttall/mountaincar-curiosity


## Required Libraries

This code has only been tested using Python 3.8. Python versions prior to 3.6 definitely won't work.

You'll need the following Python libraries. They can all be installed using the pip command.

 - tensorflow
 - gym
 - matplotlib
 - numpy
 - pillow

Additionally, some of the plotting code makes use of the following external software:

 - ffmpeg
 - imagemagick


## Project Structure

 - `main_training.py` contains all the code needed to build and train the model.
Running this will output the trained model and checkpoints to the `save` directory.
 - `main_plots.py` contains various functions for data analysis and plotting.
It will read the data in the `save` directory and output various plots to the `plots` directory.
 - The other python scripts (`mc_plot_circles.py`, `mc_random_policy.py` and `simulate_gradient_descent.py`) were used to produce some figures in the writeup but don't have anything to do with the main machine learning model.
