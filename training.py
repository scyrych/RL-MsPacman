import tensorflow as tf
from agent import DQN
import numpy as np
import os
import gym
from preprocess import preprocess_observation
from memory import replay_memory, epsilon_greedy, sample_memories

input_height = 88
input_width = 80
input_channels = 1
n_outputs = 9

learning_rate = 0.001
momentum = 0.95

n_steps = 4000000
training_start = 10000
training_interval = 4
save_steps = 1000
copy_steps = 10000
discount_rate = 0.99
skip_start = 90
batch_size = 50
checkpoint_path = "./pacman_dqn.ckpt"


def main():
    iteration = 0
    loss_val = np.infty
    game_length = 0
    total_max_q = 0
    mean_max_q = 0.0
    done = True
    state = []

    dqn = DQN()
    env = gym.make("MsPacman-v0")

    X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels])

    online_q_values, online_vars = dqn.create_model(X_state, "qnetwork_online")
    target_q_values, target_vars = dqn.create_model(X_state, "qnetwork_target")

    copy_ops = [target_var.assign(online_vars[var_name])
                for var_name, target_var in target_vars.items()]
    copy_online_to_target = tf.group(*copy_ops)

    X_action, global_step, loss, training_op, y = define_train_variables(online_q_values)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:

        restore_session(copy_online_to_target, init, saver, sess)

        while True:
            step = global_step.eval()
            if step >= n_steps:
                break

            iteration += 1
            print("\rIteration {}\tTraining step {}/{} ({:.1f})%\tLoss {:5f}\tMean Max-Q {:5f}   ".format(
                iteration, step, n_steps, step * 100 / n_steps, loss_val, mean_max_q), end="")

            state = skip_some_steps(done, env, state)

            done, q_values, next_state = evaluate_and_play_online_dqn(X_state, env, online_q_values, state, step)
            state = next_state

            mean_max_q = compute_statistics(done, game_length, mean_max_q, q_values, total_max_q)

            if iteration < training_start or iteration % training_interval != 0:
                continue

            loss_val = train_online_dqn(X_action, X_state, loss, sess, target_q_values, training_op, y)

            # Copy the online DQN to the target DQN
            if step % copy_steps == 0:
                copy_online_to_target.run()

            # Save model
            if step % save_steps == 0:
                saver.save(sess, checkpoint_path)


def define_train_variables(online_q_values):
    with tf.variable_scope("train"):
        X_action = tf.placeholder(tf.int32, shape=[None])
        y = tf.placeholder(tf.float32, shape=[None, 1])
        q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, n_outputs),
                                axis=1, keepdims=True)
        error = tf.abs(y - q_value)
        clipped_error = tf.clip_by_value(error, 0.0, 1.0)
        linear_error = 2 * (error - clipped_error)
        loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

        global_step = tf.Variable(0, trainable=False, name='global_step')
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
        training_op = optimizer.minimize(loss, global_step=global_step)
    return X_action, global_step, loss, training_op, y


def restore_session(copy_online_to_target, init, saver, sess):
    if os.path.isfile(checkpoint_path + ".index"):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()
        copy_online_to_target.run()


def skip_some_steps(done, env, state):
    if done:
        obs = env.reset()
        for skip in range(skip_start):
            obs, reward, done, info = env.step(0)
        state = preprocess_observation(obs)
    return state


def evaluate_and_play_online_dqn(X_state, env, online_q_values, state, step):
    # evaluate what to do
    q_values = online_q_values.eval(feed_dict={X_state: [state]})
    action = epsilon_greedy(q_values, step)

    # play the game
    obs, reward, done, info = env.step(action)
    next_state = preprocess_observation(obs)

    # memorize whats happened
    replay_memory.append((state, action, reward, next_state, 1.0 - done))

    return done, q_values, next_state


def compute_statistics(done, game_length, mean_max_q, q_values, total_max_q):
    total_max_q += q_values.max()
    game_length += 1
    if done:
        mean_max_q = total_max_q / game_length
    return mean_max_q


def train_online_dqn(X_action, X_state, loss, sess, target_q_values, training_op, y):
    # Sample memories and use the target DQN to produce the target Q-Value
    X_state_val, X_action_val, rewards, X_next_state_val, continues = (sample_memories(batch_size))
    next_q_values = target_q_values.eval(feed_dict={X_state: X_next_state_val})
    max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
    y_val = rewards + continues * discount_rate * max_next_q_values

    # Train the online DQN
    _, loss_val = sess.run([training_op, loss], feed_dict={X_state: X_state_val,
                                                           X_action: X_action_val,
                                                           y: y_val})
    return loss_val


if __name__ == '__main__':
    main()
