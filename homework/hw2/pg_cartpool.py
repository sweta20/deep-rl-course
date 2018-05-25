import gym
import tensorflow as tf
import numpy as np 
import pickle

env = gym.make('CartPole-v1')


obs_dim = env.observation_space.shape[0]
ac_dim = env.action_space.n
hid_dim = 20
gamma = 0.99
learning_rate = 0.001
batch_size = 10
max_steps_per_episode = 100
num_episodes = 100000
render = False
decay_rate = 0.99


policy = {}
policy['W1'] = np.random.randn(obs_dim, hid_dim) / np.sqrt(obs_dim) # "Xavier" initialization
policy['W2'] = np.random.randn(hid_dim, ac_dim) / np.sqrt(ac_dim)

grad_buffer = { k : np.zeros_like(v) for k,v in policy.iteritems() } 
rmsprop_cache = { k : np.zeros_like(v) for k,v in policy.iteritems() }

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def policy_forward(obs):
    h = np.matmul(obs, policy["W1"])
    h[h<0] = 0
    prob = np.matmul(h, policy["W2"])
    prob = softmax(prob)
    return prob, h

def policy_backward(epobs, eph, endlogp):
    
    dW2 = np.dot(eph.T, endlogp)
    dh = np.dot(policy["W2"], endlogp.T )
    dh[eph.T<=0] = 0
    dW1 = np.dot(epobs.T, dh.T)
    return {"W1": dW1, "W2": dW2}

def discounted_reward(r, gamma):
    q = np.zeros_like(r)
    q_run = 0
    for i in reversed(range(0, len(r))):
        q_run = q_run * gamma + r[i]
        q[i] = q_run
    return np.array(q)

running_reward = None

for episode in range(num_episodes):
    
    observations, dlogp, rewards, hidden = [], [], [], []
    reward_sum = 0
    obs = env.reset()
    
    for t in range(max_steps_per_episode):
        prob, h = policy_forward(obs)

        observations.append(obs)
        hidden.append(h)

        sample_ac = np.random.choice(ac_dim, p=prob)
        y = np.zeros_like(prob)
        y[sample_ac] = 1
        
        dlogp.append(y - prob)

        obs, rew, done, _ = env.step(sample_ac)
        if render:
            env.render()
            
        rewards.append(rew)
        reward_sum += rew

        if done:
            break
 
    observations = np.vstack(observations)
    hidden = np.vstack(hidden)
    dlogp = np.vstack(dlogp)

    discounted_rew = discounted_reward(rewards, gamma).reshape(-1, 1)
    discounted_rew -= np.mean(discounted_rew)
    discounted_rew /= np.std(discounted_rew)

    dlogp *= discounted_rew

    grad = policy_backward(observations, hidden, dlogp)
    for k in policy: 
        grad_buffer[k] += grad[k]

    if episode%batch_size == 0:
        for k, v in policy.iteritems():
            g = grad_buffer[k]
            rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
            policy[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
            grad_buffer[k] = np.zeros_like(v)

    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    
    if episode % 100 == 0:
        print('Episode {}\tLength: {:5d}\tEpisode Reward: {:.2f}\tAverage Reward: {:.2f}'.format(
                episode, (t + 1), reward_sum, running_reward))
        pickle.dump(policy, open('model.p', 'wb'))

    if running_reward > 99:

        obs = env.reset()
        render = True
        while(True):
            prob, h = policy_forward(obs)
            sample_ac = np.random.choice(ac_dim, p=prob)
            obs, rew, done, _ = env.step(sample_ac)
            if render:
                env.render()
            if done:
                break

        print "Problem Solved!!"
        break

