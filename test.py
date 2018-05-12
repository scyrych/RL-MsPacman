import tensorflow as tf
from agent import DQN
import gym
import time
from preprocess import preprocess_observation
import numpy as np

input_height = 88
input_width = 80
input_channels = 1
n_outputs = 9

n_max_steps = 1000000
checkpoint_path = "./pacman_dqn.ckpt"


def test_model(model_path, max_steps):
    dqn = DQN()
    env = gym.make("MsPacman-v0")

    X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels])
    online_q_values, online_vars = dqn.create_model(X_state, "qnetwork_online")
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, model_path)

        obs = env.reset()

        for step in range(max_steps):
            state = preprocess_observation(obs)

            # evaluates what to do
            q_values = online_q_values.eval(feed_dict={X_state: [state]})
            action = np.argmax(q_values)

            # plays the game
            obs, reward, done, info = env.step(action)
            env.render()
            time.sleep(0.05)
            if done:
                break
    env.close()


if __name__ == '__main__':
    test_model(checkpoint_path, n_max_steps)
