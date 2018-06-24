import tensorflow as tf
import numpy as np

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

def normalize(data, mean, std):
    return (data-mean)/ (std+1e-10)

def denormalize(data, mean, std):
    return data*(std+1e-10) + mean

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]

        self.input_state = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
        self.action = tf.placeholder(shape=[None, ac_dim], name="act", dtype=tf.float32)
        self.delta = tf.placeholder(shape=[None, ob_dim], name="delta", dtype=tf.float32)

        self.model = build_mlp(tf.concat([self.input_state, self.action], axis=1), ob_dim, "NNDynamicsModel", n_layers, size, activation, output_activation)

        self.normalization = normalization

        self.loss = tf.losses.mean_squared_error(labels=self.delta, predictions=self.model)
        self.update = tf.train.AdamOptimizer().minimize(self.loss)
        self.batch_size = batch_size
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.sess = sess

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """
        """YOUR CODE HERE """
        mean_obs, std_obs, mean_delta, std_delta, mean_action, std_action = self.normalization
        normalized_obs = normalize(data["observations"], mean_obs, std_obs)
        normalized_acs = normalize(data["actions"], mean_action, std_action)
        normalized_deltas = normalize(data["next_observations"] - data["observations"], mean_delta, std_delta)

        dataset = tf.data.Dataset.from_tensor_slices((normalized_obs, normalized_acs, normalized_deltas)).repeat().batch(self.batch_size)
        dataset_iterator = dataset.make_one_shot_iterator()
        next_element = dataset_iterator.get_next()
        loss_val = None
        for epoch in range(self.iterations):
            if epoch % 10 == 0: print("Epoch {}/{}: Loss {}".format(epoch, self.iterations, loss_val))
            for i in range(len(normalized_deltas)//self.batch_size):
                batch_obs, batch_acs, batch_deltas = self.sess.run(dataset_iterator.get_next())
                _, loss_val = self.sess.run([self.update, self.loss], feed_dict={
                                                            self.input_state: batch_obs,
                                                            self.action: batch_acs,
                                                            self.delta: batch_deltas})



    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        mean_obs, std_obs, mean_delta, std_delta, mean_action, std_action = self.normalization
        deltas = self.sess.run(self.model, feed_dict={
                                               self.input_state: normalize(states, mean_obs, std_obs),
                                               self.action: normalize(actions, mean_action, std_action)})
        
        next_states = states + denormalize(deltas, mean_delta, std_delta)

        return next_states


