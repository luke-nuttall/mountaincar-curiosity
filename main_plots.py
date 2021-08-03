#!/usr/bin/python3

import pickle
from pathlib import Path
import json
from typing import Iterable, List
import time

from PIL import Image
from matplotlib import ticker, patches, pyplot as plt
import numpy as np
import main_training as training
import subprocess


# This is a very ugly hack to fix a problem with unpickling objects which were pickled from __main__
# Because the objects were defined in __main__ when they were pickled, that's where pickle expects them to be located
Transition = training.Transition
ReplayBuffer = training.ReplayBuffer


def plot_q_evolution(purge_cache=False):
    # This code uses some single letter abbreviations for brevity
    # d = decision map, n = no thrust, l = thrust left, r = thrust right
    # This code also makes heavy use of *arr and **dict unpacking
    # f(*arr) is used to pass the values of the list 'arr' to the function 'f' as positional arguments
    # f(**dict) is used to pass the key, value pairs of 'dict' to 'f' as keyword arguments
    resolution_factor = 5  # sets the quality of the resulting images, higher = better but slower
    xstep = 0.1 / resolution_factor
    ystep = 0.01 / resolution_factor
    xs = np.arange(-1.2, 0.6, xstep)
    ys = np.arange(-0.07, 0.07, ystep)
    imshow_args = {"origin": "lower",
                   "extent": (xs[0] - xstep / 2, xs[-1] + xstep / 2, ys[0] - ystep / 2, ys[-1] + ystep / 2)}

    cache_path = Path("plots/maps/data_q.pickle")
    cache_invalid = not cache_path.exists()
    if not cache_invalid:
        # if the cache was created before the checkpoint metadata, it must be out of date
        ctime_checkpoint = training.checkpoint_meta_path.stat().st_ctime
        ctime_cache = cache_path.stat().st_ctime
        if ctime_checkpoint > ctime_cache:
            cache_invalid = True
    if purge_cache or cache_invalid:
        env, agent = training.create_env_and_agent()
        with training.checkpoint_meta_path.open("r") as fp:
            save_meta = json.load(fp)
        n_epochs = len(save_meta)

        # Here we iterate through all the model checkpoints, load each one, and
        # record the Q values calculated by the agent for each (x,v) coordinate in phase space
        data_q = np.zeros((n_epochs, len(ys), len(xs), 3), dtype=float)
        for ee, epoch in enumerate(save_meta):
            print(f"\rProbing agent at epoch {ee+1} of {n_epochs}", end="", flush=True)
            agent.load(Path(training.checkpoint_save_path + f"{ee}"))
            for ii, pos in enumerate(xs):
                for jj, vel in enumerate(ys):
                    obs = np.array([pos, vel])
                    q_values = agent.get_action_raw(obs)
                    data_q[ee, jj, ii] = q_values
        print()  # end of output line

        with cache_path.open("wb") as fp:
            pickle.dump(data_q, fp)
    else:
        with cache_path.open("rb") as fp:
            data_q = pickle.load(fp)
        n_epochs = len(data_q)

    # put the Q data into a format suitable for plotting
    # we also calculate the global min and max values here so that all the plots can use the same scale
    # this is important when we come to turn it into an animation
    data_qn = data_q[:, :, :, 1]
    data_ql = data_q[:, :, :, 0] - data_qn
    data_qr = data_q[:, :, :, 2] - data_qn
    vlim_n = {"vmin": np.min(data_qn[-1]),
              "vmax": np.max(data_qn[-1])}
    vlim_lr = {"vmin": min(np.min(data_ql[-1]), np.min(data_qr[-1])),
               "vmax": max(np.max(data_ql[-1]), np.max(data_qr[-1]))}

    # generate the images for the decision map plot
    contrast = np.abs(data_qr - data_ql)  # difference between Qs for right and left thrust
    contrast /= np.max(contrast)  # normalize it
    lightness = 1 - contrast  # we're going to fade to white as the difference between L and R goes to zero
    lightness *= 0.5  # set this to less than 1.0 to make the decision boundaries more visible
    lightness = lightness[:, :, :, np.newaxis]  # reshape it so that it has the right number of dimensions
    image = np.tile(lightness, (1, 1, 1, 3))  # repeat the values across RGB channels
    decision = np.argmax(data_q, axis=3)
    # https://stackoverflow.com/questions/36315762/indexing-in-numpy-related-to-max-argmax
    indices = tuple(np.indices(image.shape[:-1])) + (decision,)
    image[indices] = 1.0  # set the corresponding color to full brightness

    # configure how the text is displayed on all four subplots
    text_xy = (-1.15, 0.058)
    text_args = {"fontsize": 12,
                 "bbox": {"color": "white", "alpha": 0.8}}

    for ee in range(n_epochs):
        print(f"\rRendering fig {ee+1} of {n_epochs}", end="", flush=True)

        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(nrows=3, ncols=4,
                              left=0.08, right=0.93, bottom=0.06, top=0.95,
                              wspace=0.05, hspace=0.03,
                              height_ratios=(0.1, 1, 1),
                              width_ratios=(1, 1, 0.00, 0.05))
        ax_top = fig.add_subplot(gs[0, :2])
        ax_d = fig.add_subplot(gs[1, 0])
        ax_n = fig.add_subplot(gs[1, 1])
        ax_l = fig.add_subplot(gs[2, 0])
        ax_r = fig.add_subplot(gs[2, 1])
        ax_cbn = fig.add_subplot(gs[1, 3])
        ax_cblr = fig.add_subplot(gs[2, 3])

        ax_top: plt.Axes
        ax_top.set_axis_off()
        ax_top.plot([0, n_epochs-1], [0, 0], color="C1", alpha=0.4, linewidth=5)
        ax_top.scatter([ee], [0], color="C1", s=50)
        ax_top.set_title(f"Epoch {ee+1}")

        ax_d.imshow(image[ee], **imshow_args)
        ax_d.set_aspect("auto")
        ax_d.xaxis.set_ticklabels([])
        ax_d.set_ylabel("Velocity")
        ax_d.text(*text_xy, "(a) Decision Map", **text_args)
        ax_d.legend([patches.Patch(facecolor="red"), patches.Patch(facecolor="green"), patches.Patch(facecolor="blue")],
                    ["Left thrust", "No thrust", "Right thrust"], loc="lower right")

        im = ax_n.imshow(data_qn[ee], **imshow_args, cmap="viridis", **vlim_n)
        ax_n.set_aspect("auto")
        ax_n.xaxis.set_ticklabels([])
        ax_n.yaxis.set_ticklabels([])
        ax_n.text(*text_xy, "(b) $Q$ (no thrust)", **text_args)

        plt.colorbar(im, cax=ax_cbn, label="Absolute Q value")

        im = ax_l.imshow(data_ql[ee], **imshow_args, cmap="inferno", **vlim_lr)
        ax_l.set_aspect("auto")
        plt.colorbar(im, cax=ax_cblr, label="Delta Q relative to (b)")
        ax_l.set_xlabel("Position")
        ax_l.set_ylabel("Velocity")
        ax_l.text(*text_xy, "(c) $\Delta Q$ (left)", **text_args)

        im = ax_r.imshow(data_qr[ee], **imshow_args, cmap="inferno", **vlim_lr)
        ax_r.set_aspect("auto")
        ax_r.set_xlabel("Position")
        ax_r.yaxis.set_ticklabels([])
        ax_r.text(*text_xy, "(d) $\Delta Q$ (right)", **text_args)

        fig.savefig(f"plots/maps/{ee}.png", dpi=150)
        plt.close(fig)  # this prevents a memory leak because matplotlib retains all open figures in memory
    print()

    filenames = (f"{ee}.png" for ee in range(n_epochs))

    # The agent learns quickly early on and then more slowly as training progresses.
    # Therefore we'll slow down the early part of the video where things are changing quickly
    # and speed up towards the end when everything is pretty much static.
    durations = [max(0.05, 0.2 * np.exp(-ee / 100)) for ee in range(n_epochs)]
    durations[-1] = 1.0  # Also make the last frame longer, otherwise Vimeo cuts the end of the video off.

    images_to_video(Path("plots/maps"), filenames, durations, "plots/q_map.webm")


def images_to_video(folder: Path, filenames: Iterable[str], durations: Iterable[float], output: str):
    print("Generating video from images...")
    # see https://trac.ffmpeg.org/wiki/Concatenate
    # we build a text file listing all the input images and durations, then pass that file as input to ffmpeg
    filelist = folder/"filelist.txt"
    with filelist.open("w") as fp:
        for name, duration in zip(filenames, durations):
            fp.write(f"file '{name}'\nduration {duration:0.3f}\n")
        # write the final frame again at the end because otherwise the final duration is ignored
        # see https://stackoverflow.com/questions/46952350/ffmpeg-concat-demuxer-with-duration-filter-issue
        fp.write(f"file '{name}'")
    base_args = ["ffmpeg", "-f", "concat", "-i", str(filelist),
                 "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "25", "-vf", "fps=25", "-y"]
    # it's best to use a 2-pass encoding process to achieve maximum compression
    subprocess.run(base_args + ["-pass", "1", "-f", "null", "/dev/null"])
    subprocess.run(base_args + ["-pass", "2", output])


def images_to_gif(folder: Path, filenames: Iterable[str], durations: Iterable[float], output: str, resize=None):
    print("Generating GIF from images...")
    args = ["convert"]  # `convert` is part of the imagemagick suite. See `man convert` for docs.
    for filename, duration in zip(filenames, durations):
        args += ["-delay", str(int(duration*100)), str(folder / filename)]
    args += ["-loop", "0"]  # number of times to loop the animation. 0 is infinite.
    args += ["-layers", "Optimize"]  # reduces file size
    if resize is not None:
        args += ["-resize", resize]
    if not output.endswith(".gif"):
        output += ".gif"  # make sure the output is actually a gif and not some other image format
        # could alternatively use an output format prefix, i.e.    output = "GIF:" + output
        # https://legacy.imagemagick.org/Usage/files/#save
    args += [output]
    subprocess.run(args)
    # could try passing the output through gifsicle to further reduce the file size


def env_save_image(env, filename: str):
    """
    Renders the environment in its current state and saves that rendered image as `filename`.
    Note that the format will be determined from the filename extension.
    """
    rgb = env.render(mode="rgb_array")
    img = Image.fromarray(rgb)
    img.save(filename)


def render_gameplay_video():
    env, agent = training.create_env_and_agent()
    with training.checkpoint_meta_path.open("r") as fp:
        save_meta = json.load(fp)
    best_epoch = np.argmax([epoch['reward'] for epoch in save_meta])
    best_reward = save_meta[best_epoch]['reward']
    print(f"Loading agent from epoch {best_epoch} (reward={best_reward})")
    agent.load(Path(f"{training.checkpoint_save_path}{best_epoch}"))

    fps = env.metadata.get('video.frames_per_second', 30)
    frametime = 1 / fps
    print(f"FPS: {fps}")

    done = False
    step = 0
    obs = env.reset()
    env.render()
    env_save_image(env, f"plots/video_frames/{step}.png")
    filenames = [f"{step}.png"]
    durations = [frametime]
    last_time = time.time()
    while not done:
        action = agent.get_action(obs, 0.0)
        obs, reward, done, _ = env.step(action)
        step += 1
        env_save_image(env, f"plots/video_frames/{step}.png")
        filenames.append(f"{step}.png")
        durations.append(frametime)
        # limit the FPS for display
        now = time.time()
        dt = now - last_time
        if dt < frametime:
            time.sleep(frametime - dt)
        last_time = now

    # Make the last frame longer. This way it looks better when looping.
    durations[-1] = 1.0

    images_to_gif(Path("plots/video_frames"), filenames, durations, "plots/gameplay.gif")


def plot_buffer_tracks(buffer=None, outpath=None):
    if buffer is None:
        buffer = training.ReplayBuffer.load(training.buffer_save_path)
    fig, ax = plt.subplots()
    ax: plt.Axes
    xs, ys = [], []
    for sample in buffer.experience:
        if (-0.6 <= sample.obs1[0] <= -0.4) and (sample.obs1[1] == 0):
            ax.plot(xs, ys, color="C0", alpha=0.1, zorder=0)
            xs = [sample.obs1[0]]
            ys = [sample.obs1[1]]
        xs.append(sample.obs2[0])
        ys.append(sample.obs2[1])
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_xlim(-1.2, 0.6)
    ax.set_ylim(-0.07, 0.07)

    fig.tight_layout()
    if outpath is None:
        outpath = "plots/buffer_tracks.png"

    fig.savefig(outpath, dpi=150)


def plot_q_final():
    resolution_factor = 10  # sets the quality of the resulting images, higher = better but slower
    xstep = 0.1 / resolution_factor
    ystep = 0.01 / resolution_factor
    xs = np.arange(-1.2, 0.6, xstep)
    ys = np.arange(-0.07, 0.07, ystep)
    imshow_args = {"origin": "lower",
                   "extent": (xs[0] - xstep / 2, xs[-1] + xstep / 2, ys[0] - ystep / 2, ys[-1] + ystep / 2)}

    env, agent = training.create_env_and_agent()
    with training.checkpoint_meta_path.open("r") as fp:
        save_meta = json.load(fp)
    n_epochs = len(save_meta)

    # Here we iterate through all the model checkpoints, load each one, and
    # record the Q values calculated by the agent for each (x,v) coordinate in phase space
    data_q = np.zeros((len(ys), len(xs), 3), dtype=float)
    ee = n_epochs - 1
    epoch = save_meta[ee]
    print(f"\rProbing agent at epoch {ee + 1} of {n_epochs}")
    agent.load(Path(training.checkpoint_save_path + f"{ee}"))
    for ii, pos in enumerate(xs):
        for jj, vel in enumerate(ys):
            obs = np.array([pos, vel])
            q_values = agent.get_action_raw(obs)
            data_q[jj, ii] = q_values

    fig, ax = plt.subplots()
    im = ax.imshow(data_q[:, :, 1], **imshow_args)
    ax.set_aspect("auto")
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    plt.colorbar(im, label="Q (no thrust)")
    plt.tight_layout()
    fig.savefig(f"plots/q_nothrust.png", dpi=150)


def plot_feature_space():
    xstep = 0.05
    ystep = 0.005
    xs = np.arange(-1.2, 0.6, xstep)
    ys = np.arange(-0.07, 0.07, ystep)

    env, agent = training.create_env_and_agent()
    with training.checkpoint_meta_path.open("r") as fp:
        save_meta = json.load(fp)
    n_epochs = len(save_meta)

    ee = n_epochs - 1
    print(f"\rProbing agent at epoch {ee + 1} of {n_epochs}")
    agent.load(Path(training.checkpoint_save_path + f"{ee}"))

    features = np.zeros([len(ys), len(xs), 2], dtype=float)
    for ii, pos in enumerate(xs):
        for jj, vel in enumerate(ys):
            obs = np.array([pos, vel])
            features[jj, ii] = agent.model_features(np.atleast_2d(obs.astype('float32')))[0]

    fig, (ax1, ax2) = plt.subplots(ncols=2)

    extent = [xs[0]-xstep/2, xs[-1]+xstep/2, ys[0]-ystep/2, ys[-1]+ystep/2]

    im1 = ax1.imshow(features[:, :, 0], origin="lower", extent=extent, cmap="viridis")
    ax1.set_aspect("auto")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(features[:, :, 1], origin="lower", extent=extent, cmap="viridis")
    ax2.set_aspect("auto")
    plt.colorbar(im2, ax=ax2)

    plt.show()


def populate_buffer_at_epochs(epochs: List[int], n_per_epoch: int, epsilon=0.2):
    # creates a replay buffer and fills it with data by running agents from each of the specified epochs
    # epsilon is the randomness factor
    env, agent = training.create_env_and_agent()
    buf = training.ReplayBuffer(n_per_epoch * len(epochs))
    for epoch in epochs:
        agent.load(Path(training.checkpoint_save_path + f"{epoch}"))
        training.collect_data(env, agent, buf, epsilon, n_per_epoch)
    return buf


def plot_learning_rate():
    with training.checkpoint_meta_path.open("r") as fp:
        save_meta = json.load(fp)

    fig, ax = plt.subplots(figsize=(8, 4))

    xlim = (-1, 200)

    ax: plt.Axes
    ax.plot([epoch["learn_rate"] for epoch in save_meta])
    ax.semilogy()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_xlim(*xlim)

    fig.tight_layout()
    fig.savefig(f"plots/learning_rate.png", dpi=150)


def plot_q_loss_and_reward():
    with training.checkpoint_meta_path.open("r") as fp:
        save_meta = json.load(fp)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    xlim = (-1, 200)

    ax1.plot([epoch["loss_q"] for epoch in save_meta])
    ax1.semilogy()
    ax1.set_ylabel("Q Loss")
    ax1.xaxis.set_ticklabels([])
    ax1.set_xlim(*xlim)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

    ax2.plot([epoch["reward"] for epoch in save_meta])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Avg. Reward")
    ax2.set_xlim(*xlim)

    fig.tight_layout()
    fig.savefig(f"plots/q_loss_reward.png", dpi=150)


def plot_forward_loss():
    with training.checkpoint_meta_path.open("r") as fp:
        save_meta = json.load(fp)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    xlim = (-1, 200)

    ax.plot([epoch["loss_forward"] for epoch in save_meta])
    ax.semilogy()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Forward Loss")
    ax.set_xlim(*xlim)

    fig.tight_layout()
    fig.savefig(f"plots/forward_loss.png", dpi=150)


def main():
    # comment/uncomment lines here to select which plots to generate
    plot_buffer_tracks()
    plot_q_evolution()
    render_gameplay_video()
    plot_q_final()
    plot_feature_space()
    plot_learning_rate()
    plot_q_loss_and_reward()
    plot_forward_loss()

    # The code below is for investigating the sudden increase in forward loss around epoch 17
    r1 = list(range(12, 17))
    r2 = list(range(17, 22))
    buf1 = populate_buffer_at_epochs(r1, 20_000)
    buf2 = populate_buffer_at_epochs(r2, 20_000)
    plot_buffer_tracks(buf1, f"plots/tracks_epoch_{r1[0]}-{r1[-1]}.png")
    plot_buffer_tracks(buf2, f"plots/tracks_epoch_{r2[0]}-{r2[-1]}.png")


if __name__ == "__main__":
    main()
