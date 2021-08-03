#!/usr/bin/python3

from typing import List
import numpy as np
import tensorflow as tf
import gym
from pathlib import Path
from collections import namedtuple
import pickle
import json


checkpoint_save_path = "save/checkpoint/ckpt"
buffer_save_path = Path("save/replay_buffer.pickle")
checkpoint_meta_path = Path("save/checkpoint_meta.json")
reward_save_path = Path("save/reward_data.json")


# Represents a single state transition consisting of an initial observation, action, final observation, and reward.
# An episode of gameplay is made up of a sequence of these transitions.
Transition = namedtuple("Transition", ["obs1", "action", "obs2", "reward", "done"])


class ReplayBuffer:
    """
    Used to store a history of Transition objects from which training data can be randomly sampled.
    Randomly sampling from a replay buffer helps to prevent correlations between samples, improving training.

    Internally this is implemented as a circular buffer.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity  # the maximum number of records the buffer can hold
        self.size = 0  # the number of records currently in the buffer
        self.ptr = 0  # the index of the oldest record in the buffer (used for overwriting)
        self.experience: List[Transition] = []

    def add(self, sample: Transition):
        """
        Stores a single transition in the buffer.
        """
        # if the buffer isn't full, just append
        # appending to a list is asymptotically O(1)
        if self.size < self.capacity:
            self.experience.append(sample)
            self.size += 1
        # otherwise we overwrite the oldest value and increment the pointer
        # if we implemented it as pop(0) followed by append(sample) then it would take O(n) time
        # doing it this way is always O(1)
        else:
            self.experience[self.ptr] = sample
            self.ptr = (self.ptr + 1) % self.capacity

    def random_sample(self, n: int) -> List[Transition]:
        """
        Returns a list of `n` transitions randomly sampled from the replay buffer.
        """
        assert self.size > 0, "You can't sample from an empty ReplayBuffer."
        indices = np.random.randint(low=0, high=self.size, size=n)
        return [self.experience[ii] for ii in indices]

    def save(self, path: Path):
        """
        Writes the buffer to a file on disk. Use the `load()` classmethod to restore a saved buffer.

        :param path: the file which data should be written to.
        """
        with path.open("wb") as fp:
            pickle.dump(self, fp)

    @classmethod
    def load(cls, path: Path):
        """
        Creates a new buffer using data from a file previously created by the `save()` method.

        :param path: the location of the file
        :return: the new buffer
        """
        with path.open("rb") as fp:
            self = pickle.load(fp)
        assert isinstance(self, cls), "Tried to load() an object which is not of type ReplayBuffer."
        return self


def build_net(inputs: List[int], hidden_units: List[int], output: int, out_act: str, name=None):
    # inputs
    if len(inputs) > 1:
        layers_in = [tf.keras.Input(shape=(n,)) for n in inputs]
        layer = tf.keras.layers.concatenate(layers_in)
    else:
        layer = tf.keras.Input(shape=(inputs[0],))
        layers_in = [layer]

    # hidden layers
    for nn in hidden_units:
        layer = tf.keras.layers.Dense(nn, activation='relu')(layer)

    # output
    layer_out = tf.keras.layers.Dense(output, activation=out_act)(layer)
    return tf.keras.Model(inputs=layers_in, outputs=layer_out, name=name)


class Agent:
    def __init__(self, n_states: int, n_actions: int, hidden_q: List[int], hidden_features: List[int],
                 hidden_others: List[int], gamma: float):
        # for a more complex problem we'd use a smaller feature space than the observation space,
        # but for MountainCar there are only two observable variables so we'll keep the size the same
        n_features = n_states
        # The main Q network which will be trained to predict the expected reward from each action in a given state.
        self.model_q = build_net([n_states], hidden_q, n_actions, "linear", "Q_net")
        # A copy of the main Q network which only get synchronised periodically.
        # Used for estimating the right hand side of the Bellman equation.
        self.model_target = build_net([n_states], hidden_q, n_actions, "linear", "Target_net")
        # The feature network which maps observations into a more useful space.
        self.model_features = build_net([n_states], hidden_features, n_features, "tanh", "Feature_net")
        # The forward model tries to predict the next feature vector from the previous one and the action.
        self.model_forward = build_net([n_features, n_actions], hidden_others, n_features, "tanh", "Forward_net")
        # The inverse network tries to predict the action given the initial and final features.
        self.model_inverse = build_net([n_features, n_features], hidden_others, n_actions, "softmax", "Inverse_net")

        # Each model needs its own optimiser because the optimisers we're using are stateful.
        # They all share the same learning rate for now, but we could separate those out too.
        # Using a tf.Variable for the learning rate lets us change it during training
        self.lr = tf.Variable(1e-3)  # 1e-3 is the default value for the Adam optimizer in tensorflow
        self.optimizer_q = tf.optimizers.Adam(self.lr)
        self.optimizer_features = tf.optimizers.Adam(self.lr)
        self.optimizer_forward = tf.optimizers.Adam(self.lr)
        self.optimizer_inverse = tf.optimizers.Adam(self.lr)

        self.gamma = gamma  # a discounting factor for future rewards in the Bellman equation
        self.n_actions = n_actions
        self.intrinsic_reward_scale = tf.Variable(100, dtype=float)  # adjust the strength of the intrinsic reward

        # Here we define what gets saved by the save() function.
        self.checkpoint = tf.train.Checkpoint(
            model_q=self.model_q,
            model_features=self.model_features,
            model_forward=self.model_forward,
            model_inverse=self.model_inverse,
            optimizer_q=self.optimizer_q,
            optimizer_features=self.optimizer_features,
            optimizer_forward=self.optimizer_forward,
            optimizer_inverse=self.optimizer_inverse,
        )

    def get_action(self, observation, epsilon: float):
        """
        Returns the action to take given an observation.
        :param observation: The observation data provided by the environment.
        :param epsilon: A randomness factor between 0 and 1.
            0 means the agent behaves deterministically, always taking the action it believes will give the best reward.
            1 means the agent will choose an action completely at random.
        """
        if np.random.random() < epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.model_q(np.atleast_2d(observation.astype('float32')))[0])

    def get_action_raw(self, observation):
        """
        Returns the raw output of the Q network for the given input.
        :param observation: The observation data provided by the environment.
        """
        return self.model_q(np.atleast_2d(observation.astype('float32')))[0]

    def update_target(self):
        """
        Copies the weights from the training Q network into the target Q network.
        """
        self.model_target.set_weights(self.model_q.get_weights())

    @tf.function
    def train_step(self, obs1, actions, obs2, rewards, dones):
        """
        Performs a single step of stochastic gradient descent on each of the agent's models.
        All inputs must be TF tensors.
        """
        acts = tf.one_hot(actions, self.n_actions)

        if self.intrinsic_reward_scale > tf.constant(0.0):
            # ---------------------------------------------------------------
            # TRAINING THE OBSERVATION->FEATURE AND INVERSE KINEMATICS MODELS
            # ---------------------------------------------------------------

            # try to predict the action given the previous and next observations
            with tf.GradientTape(persistent=True) as tape:
                features1 = self.model_features(obs1)
                features2 = self.model_features(obs2)
                acts_pred = self.model_inverse((features1, features2))
                loss_inverse = tf.keras.losses.categorical_crossentropy(acts, acts_pred)

            # train the features model
            vars_feat = self.model_features.trainable_variables
            grad_feat = tape.gradient(loss_inverse, vars_feat)
            self.optimizer_features.apply_gradients(zip(grad_feat, vars_feat))

            # train the inverse model
            vars_inv = self.model_inverse.trainable_variables
            grad_inv = tape.gradient(loss_inverse, vars_inv)
            self.optimizer_inverse.apply_gradients(zip(grad_inv, vars_inv))

            # ---------------------------------------------------------------
            # TRAINING THE FORWARD KINEMATICS MODEL AND CALCULATING CURIOSITY
            # ---------------------------------------------------------------

            # try to predict the next observation given the previous one and the action taken
            with tf.GradientTape() as tape:
                features2_pred = self.model_forward((features1, acts))
                loss_forward = tf.keras.losses.mean_squared_error(features2, features2_pred)

            # train the forward model
            vars_fw = self.model_forward.trainable_variables
            grad_fw = tape.gradient(loss_forward, vars_fw)
            self.optimizer_forward.apply_gradients(zip(grad_fw, vars_fw))

            # use the forward loss as an intrinsic curiosity reward signal
            rewards_adjusted = rewards + loss_forward * self.intrinsic_reward_scale
        else:
            rewards_adjusted = rewards
            loss_inverse = tf.constant(0.0)
            loss_forward = tf.constant(0.0)

        # --------------------
        # TRAINING THE Q MODEL
        # --------------------

        # Bellman optimality equation:
        # Q(s1, a1) = Expectation[reward + gamma * max(Q(s2, a2))]

        # The estimated value of the best action which could be taken in the next timestep (starting from obs2)
        val_next = tf.math.reduce_max(self.model_target(obs2), axis=1)
        # The reward in this timestep, plus the discounted value from the next timestep (unless done==True)
        val_actions = rewards_adjusted + (1 - dones) * self.gamma * val_next

        # now we use the Q network to try to predict the value of the action which was taken
        with tf.GradientTape() as tape:
            # this is basically an inner (dot) product between the vector of action Qs
            # and the one-hot vector containing the action(s) taken in the remembered experience
            val_pred = tf.math.reduce_sum(self.model_q(obs1) * acts, axis=1)
            loss_q = tf.math.reduce_mean(tf.square(val_actions - val_pred))  # mean squared error

        vars_q = self.model_q.trainable_variables
        grad_q = tape.gradient(loss_q, vars_q)
        # grad_q = [tf.clip_by_value(grad, -1., 1.) for grad in grad_q]  # clip the gradients to prevent explosions
        self.optimizer_q.apply_gradients(zip(grad_q, vars_q))

        return loss_inverse, loss_forward, loss_q

    def train(self, buffer: ReplayBuffer, batch_size: int):
        """
        Performs a training step using data sampled from a ReplayBuffer.
        This function does the necessary conversion from python datatypes to TF tensors before calling train_step()
        :param buffer: The ReplayBuffer to sample from.
        :param batch_size: The number of samples to use for each weight update step.
        """
        batch = buffer.random_sample(batch_size)

        obs1 = tf.constant([step.obs1 for step in batch], dtype=tf.float32)
        actions = tf.constant([step.action for step in batch], dtype=tf.int32)
        obs2 = tf.constant([step.obs2 for step in batch], dtype=tf.float32)
        rewards = tf.constant([step.reward for step in batch], dtype=tf.float32)
        dones = tf.constant([step.done for step in batch], dtype=tf.float32)

        loss_inv, loss_fw, loss_q = self.train_step(obs1, actions, obs2, rewards, dones)

        avg_loss_q = loss_q.numpy().mean()
        avg_loss_fw = loss_fw.numpy().mean()
        avg_loss_inv = loss_inv.numpy().mean()

        return avg_loss_q, avg_loss_fw, avg_loss_inv

    def save(self, path: Path):
        """
        Stores a copy of the current model weights and optimiser states.
        :param path: The prefix used for all the files. It can't be a directory.
        """
        assert not path.is_dir(), f"Unable to save model with prefix {path}. It is a directory."
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint.write(str(path))

    def load(self, path: Path):
        """
        Loads the weights and optimiser states from files created using the save() function.
        :param path: The same file prefix which was previously passed to save().
        """
        self.checkpoint.read(str(path)).expect_partial()
        self.update_target()

    def update_learning_rate(self, new_lr):
        self.lr.assign(new_lr)

    def update_intrinsic_reward(self, new_reward_scale):
        self.intrinsic_reward_scale.assign(new_reward_scale)

    def save_checkpoint(self, path: Path):
        """
        Stores a copy of the current model weights and optimiser states.
        This method differs from save() in that it automatically numbers the checkpoints instead of overwriting.
        :param path: The prefix used for all the files. It can't be a directory.
        """
        assert not path.is_dir(), f"Unable to save model with prefix {path}. It is a directory."
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint.save(str(path))

    def load_checkpoint(self, path: Path):
        """
        Loads the weights and optimiser states from files created using the save_checkpoint() function.
        :param path: The same file prefix which was previously passed to save().
        """
        self.checkpoint.restore(str(path))
        self.update_target()


def collect_data(env, agent, buffer: ReplayBuffer, epsilon: float, n_steps: int):
    obs = env.reset()
    for ii in range(n_steps):
        action = agent.get_action(obs, epsilon)
        next_obs, reward, done, info = env.step(action)
        buffer.add(Transition(obs, action, next_obs, reward, done))
        if done:
            obs = env.reset()
        else:
            obs = next_obs


def evaluate(env, agent, n_rounds: int) -> float:
    rewards = 0
    for _ in range(n_rounds):
        obs = env.reset()
        done = False
        while not done:
            action = agent.get_action(obs, 0.0)
            obs, reward, done, info = env.step(action)
            rewards += reward
    return rewards / n_rounds


def create_env_and_agent():
    env = gym.make('MountainCar-v0')
    n_states = len(env.observation_space.sample())
    n_actions = env.action_space.n
    hidden_q = [128, 128]
    hidden_features = [4, 4]
    hidden_others = [32, 32]
    gamma = 0.99  # discounting factor for future rewards in the Bellman equation
    agent = Agent(n_states, n_actions, hidden_q, hidden_features, hidden_others, gamma)
    return env, agent


def train_agent(env, agent):
    copy_period = 50  # number of training steps between updates to the target network
    buffer_capacity = 100_000  # capacity of the replay buffer
    batch_size = 32
    batches_per_epoch = 10_000  # The definition of an epoch here is a bit arbitrary
    data_collection_interval = 500  # measured in batches
    data_collection_size = 200  # number of steps to play the game for when collecting training data
    print_interval = 100  # measured in batches
    n_epochs = 200  # total number of epochs to train for

    learn_rate = 5e-3  # the initial learning rate
    lr_decay = 0.96  # The learning rate is multiplied by this value after each epoch. Must be less than 1.
    lr_min = 1e-8  # the learning rate will never go lower than this value

    buffer = ReplayBuffer(buffer_capacity)
    collect_data(env, agent, buffer, 0.5, 200)
    best_reward = -np.inf
    save_counter = 0
    save_meta = []
    for epoch in range(n_epochs):
        agent.update_learning_rate(learn_rate)

        losses_sum = np.zeros(3)
        for step in range(batches_per_epoch):
            if step % print_interval == 0:
                # this code lets us display and update the progress within a single epoch on a single line
                # "\r" returns the cursor to the start of the line, so it will overwrite what was there previously
                # end="" means that we stay on the same line rather than putting each update on a new line
                print(f"\repoch {epoch}:\tLR={learn_rate:0.2E}, BS={batch_size},\tstep {step+1}/{batches_per_epoch}", end="")
            if step % copy_period == 0:
                agent.update_target()
            if step % data_collection_interval == 0:
                collect_data(env, agent, buffer, 0.2, data_collection_size)
            losses_sum += agent.train(buffer, batch_size)
        loss_q, loss_fw, loss_inv = losses_sum / batches_per_epoch
        print(f"\repoch {epoch}:\tLR={learn_rate:0.2E}, BS={batch_size},\tstep {batches_per_epoch}/{batches_per_epoch}\t", end="")
        print(f"inverse loss {loss_inv:0.2E},\tforward loss {loss_fw:0.2E},\tQ loss {loss_q:0.2f},\t", end="")

        # evaluate the agent's performance after each epoch
        # doing 100 iterations of the game here is definitely overkill, but it lets us plot a more precise graph
        # in production code this should really be reduced to 10 or less
        reward = evaluate(env, agent, 100)
        print(f"reward {reward}", end="")

        # save the agent's state and record all the performance metrics at the end of each epoch
        agent.save(Path(checkpoint_save_path + f"{save_counter}"))
        save_meta.append({
            "reward": reward,
            "learn_rate": learn_rate,
            "loss_q": loss_q,
            "loss_forward": loss_fw,
            "loss_inverse": loss_inv,
        })
        save_counter += 1

        # let's also save the current state of the replay buffer and the training progress metadata
        buffer.save(buffer_save_path)
        with checkpoint_meta_path.open("w") as fp:
            json.dump(save_meta, fp)

        # we'll also add a note to the end of the output for this epoch if the agent improved on its performance
        if reward >= best_reward:
            best_reward = reward
            print(" (new best)", end="")

        # update the learning rate according to an exponential decay
        if learn_rate > lr_min:
            learn_rate = max(learn_rate * lr_decay, lr_min)

        print()  # end this line of output


def main():
    env, agent = create_env_and_agent()
    train_agent(env, agent)
    print("Training complete!\n")

    with checkpoint_meta_path.open("r") as fp:
        save_meta = json.load(fp)
    best_save = int(np.argmax([epoch['reward'] for epoch in save_meta]))
    best_reward = save_meta[best_save]['reward']
    print(f"Best performance at was epoch {best_save} with an average score of {best_reward}")


if __name__ == "__main__":
    main()
