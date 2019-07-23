from abc import ABC, abstractmethod
import numpy as np
import os 

import tensorflow as tf
from keras import backend as K


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


class Gems(ABC):
    """
        Abstract class for epsilon space 
    """

    def __init__(self, model, Xval, yval, FIM, logger = None):
        self.model = model 
        self.orig_weights = model.get_weights()
        self.Xval = Xval
        self.yval = yval 
        self.fisher = FIM
        self.fisher_mask = None
        self.logger = logger
        super().__init__()
    
    @abstractmethod
    def build_epsilon_space(self, num_samples):
        pass

    @abstractmethod
    def evaluate_model(self, new_weights):
        """ Calculate the loss/accuracy of a new set of weights 
        Keyword arguments: 
        new_weights: list of var/weights to test
        
        Returns: accuracy of new_weights on (self.Xval, self.yval)
        """
        self.model.set_weights(new_weights)
        loss, acc = self.model.evaluate(self.Xval, self.yval, verbose = 0)
        return acc
    
    @abstractmethod
    def restore_model(self):
        """
            Restore the model's weights to the original optimal. 
        """
        self.model.set_weights(self.orig_weights)


class SingleSphere(Gems):
    """
        Learn a spherical epsilon space over fisher parameters 
    """

    def evaluate_model(self, new_weights):
        return super().evaluate_model(new_weights)

    def restore_model(self):
        return super().restore_model()

    def sample_uniform(self, mvars, radius):
        """ 
        Uniformly sample a point from the surface of a sphere

        Keyword arguments
        mvars: list of layer weights [(w1, b1), (w2, b2)...] corresponding to sphere 
        center 
        radius: radius of sphere

        Returns: 
        point: a sampled point on the surface of sphere 
        """

        #TODO: ONLY ADD NOISE OVER FISHER WEIGHTS??
        point = []
        sum_sq = 0
        for i in range(len(mvars)):
            gauss = np.random.normal(0.0, 1.0, mvars[i].shape) # sample from standard normal gaussian 
            gauss = gauss
            point.append(gauss)
            sum_sq += np.sum(np.square(gauss))
        norm = np.sqrt(sum_sq)

        for i in range(len(point)):
            point[i] = radius*point[i] / norm + mvars[i]
        
        return point


    def build_epsilon_space(self, epsilon, num_samples, start_radius, tol = 0.02, verbose = False):
        """
        Build an epsilon space for specified epsilon, performing binary search over radii
       
        Keyword arguments 
        epsilon: cutoff accuracy 
        num_samples: number of models to sample at each radius 
        start_radius: starting radius.  

        Returns: nothing, but stores the following internal values: 
        self.ep_radius: radius for resulting epsilon space 
        """
        r_lower = 0.0 
        r_upper = start_radius*2.0
        while True:
            radius = (r_lower + r_upper) / 2.0
            if r_upper - r_lower <= tol:
               self.ep_radius = radius
               break
            increase_radius = True
            for i in range(num_samples):
                pert_weights = self.sample_uniform(self.orig_weights, radius)
                acc = self.evaluate_model(pert_weights)
                if verbose:
                    print("Iter-{:d}. Radius: {:0.02f}. Accuracy: {:0.003f}".format(i, radius, acc))
                if acc < epsilon:
                    # perturbed model lies outside epsilon space - resize sphere 
                    r_upper = radius
                    increase_radius = False
                    break 
            if increase_radius:
                r_lower = radius
        self.restore_model()
        print("Calculated spherical epsilon space with R = {:0.03f}".format(self.ep_radius))
    
    def build_epsilon_space_from_data(self, epsilon, X, y, mask):
        self.fisher_mask = mask
        distance_acc = []
        for i in range(len(X)):
            w = X[i]
            v1, v2 = w[0], w[1]
            #v1 = v1.reshape(784, 10)
            #v2 = v2.reshape(10)
            mvars = [v1, v2]
            dist = get_weight_distance(mvars, self.orig_weights)
            acc = y[i]
            distance_acc.append((dist, acc))
        for i in range(len(distance_acc)):
            if distance_acc[i][1] < epsilon:
                self.ep_radius = distance_acc[i][0]
                return
        self.ep_radius = distance_acc[i][0]
        #print("Found radius: %f" % self.ep_radius)
        
        

def get_weight_distance(weights1, weights2):
    distance = 0.0
    assert len(weights1) == len(weights2)
    for i in range(len(weights1)):
        distance  += np.sum(np.square(weights1[i] - weights2[i]))
    return np.sqrt(distance)

def l2_dist(w1, w2):
    return np.sum(np.square(w1 - w2))

def check_intersection(e1, e2):
    r1 = e1.ep_radius
    r2 = e2.ep_radius
    weight_dist = get_weight_distance(e1.orig_weights, e2.orig_weights)
    return weight_dist <= r1 + r2

def single_sphere_intersection(e1, e2):
    """ Return model in the intersection of passed epsilon space. Currently only works for two agents. 
    For any pair of parameters, there are three cases: 

        1) both parameters are unimportant (fim = 0): average the parameters 
        2) one parameter is important (fim > 0) and one is unimportant (fim = 0): set to important parameter val 
        3) both parameters are important: take weighted average according to radii 

    Keyword arguments:
    e1: SingleSphere epsilon space for first agent. 
    e2: SingleSphere epsilon space for second agent 

    Returns weights for intersected model if it exists, None else.
    """

    num_vars = len(e1.orig_weights)
    intersection_vars = []
    fisher_vars1 = e1.mask
    fisher_vars2 = e2.mask
    r1 = e1.ep_radius
    r2 = e2.ep_radius
    weight_dist = get_weight_distance(e1.orig_weights, e2.orig_weights)
    intersection = weight_dist <= r1 + r2
    print("Weight distance: {}. R1: {}. R2: {}".format(weight_dist, r1, r2))

    total_distance = 0.0
    for v in range(num_vars):
        p1 = e1.orig_weights[v]
        p2 = e2.orig_weights[v]
        f1 = fisher_vars1[v]
        f2 = fisher_vars2[v]
        weights = np.zeros_like(p1)
        
        # Case 1: f1 = f2 = 0 (both unimportant parameters)
        mask = np.logical_and(np.logical_and(np.logical_not(f1), np.logical_not(f2)), f2 == 0.0) 
        weights = 0.5*(np.multiply(mask, p1) + np.multiply(mask, p2)) 

        # Case 2a: f1 > 0, and f2 == 0
        mask = np.logical_and(f1, np.logical_not(f2))
        weights += np.multiply(mask, p1)

        # Case 2b: f1 == 0, and f2 > 0
        mask = np.logical_and(f2, np.logical_not(f1))
        weights += np.multiply(mask, p2)
        
        # Case 3: f1 > 0, f2 > 0
        z1 = r2 / (r1 + r2)
        z2 = r1 / (r1 + r2)

        mask = np.logical_and(np.logical_and(f1, f2), f2 > 0.0)
        weights += z1*np.multiply(mask, p1) + z2*np.multiply(mask, p2)
        total_distance += l2_dist(np.multiply(mask, p1), np.multiply(mask, p2)) 
        intersection_vars.append(weights)

    #print("Total distance: {:0.02f}".format(np.sqrt(total_distance)))
    return intersection_vars, intersection

def multiple_sphere_intersection(epsilon_spaces):
    weight_avg = [np.zeros_like(epsilon_spaces[0].orig_weights[0]), np.zeros_like(epsilon_spaces[0].orig_weights[1])]
    for ep_space in epsilon_spaces:
        for i in range(len(ep_space.orig_weights)):
            weight_avg[i] += ep_space.orig_weights[i]
    for i in range(len(weight_avg)):
        weight_avg[i] = weight_avg[i] / len(epsilon_spaces)

    gems_vars = []
    for i in range(len(weight_avg)):
        gems_vars.append(tf.Variable(weight_avg[i], dtype=tf.float32))

    obj = 0
    for ep_space in epsilon_spaces:
        # Get mask, optimal model, and radius from agent
        agent_vars = ep_space.orig_weights
        mask = ep_space.mask 
        r = ep_space.ep_radius
        num_vars = len(agent_vars)

        tf_mask = []
        tf_agent_vars = []
        # create tensorflow arrays
        for v in range(num_vars):
            tf_agent_vars.append(tf.constant(agent_vars[v], dtype=tf.float32))
            tf_mask.append(tf.constant(agent_vars[v], dtype=tf.float32))
            
        dist = tf.reduce_sum(tf.square(tf.multiply(tf_mask[0], tf_agent_vars[0]) - tf.multiply(tf_mask[0], gems_vars[0])))
        dist = dist + tf.reduce_sum(tf.square(tf.multiply(tf_mask[1], tf_agent_vars[1]) - tf.multiply(tf_mask[1], gems_vars[1])))
        dist = tf.sqrt(dist)
        obj += tf.maximum(0.0, dist - r)
        

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train_op = optimizer.minimize(obj)

    num_steps = 1000
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(gems_vars[0].assign(weight_avg[0]))
    sess.run(gems_vars[1].assign(weight_avg[1]))
    for i in range(num_steps):
        _, obj_i = sess.run([train_op, obj])
        #if i % 200 == 0:
        #    print(i, obj_i)
    grad = sess.run(tf.gradients(obj, gems_vars))
    #print("Grad mean:", np.mean(np.concatenate([g.ravel() for g in grad])))
    return sess.run(gems_vars), True # TODO: FIX THIS
    
    #gems_model = create_logit(intersected_weights, Xts_flat, yts_flat, bias=True)
    #loss, acc = gems_model.evaluate(Xts_flat, yts_flat, verbose=0)
    #print("Epsilon: {:0.2f}. Accuracy: {:0.03f} Loss: {:0.02f}".format(ep, acc, loss))


def get_vars(model):
    # returns a list of the TF variables for model, in the format of weights
    num_layers = len(model.layers)
    tfvars = []
    for i in range(num_layers):
        if len(model.layers[i].weights) == 0:
            continue
        for j in range(len(model.layers[i].weights)):
            tfvars.append(model.layers[i].weights[j])
    return tfvars
        
def compute_fisher(model, X, num_samples=200):
    # TODO: Move this to a more reasonable directory 
    # Returns the fisher information for each variable [(w1, b1), (w2, b2), ...]

    sess = K.get_session()
    
    # initialize Fisher information for most recent task
    tfvars = get_vars(model)
    vars_fisher = []
    for v in range(len(tfvars)):
        vars_fisher.append(np.zeros(tfvars[v].get_shape().as_list()))
    
    # sampling a random class from softmax
    probs = model.layers[-1].output
    class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])
    
    # Select random images
    img_indices = np.random.choice(range(len(X)), num_samples, replace = False)
    for im_ind in img_indices:
        
        # compute first-order derivatives
        ders = sess.run(tf.gradients(tf.log(probs[0,class_ind]), tfvars), feed_dict={model.layers[0].input: X[im_ind:im_ind+1]})
        p = sess.run(probs, feed_dict={model.layers[0].input: X[im_ind:im_ind+1]})
        
        # square the derivatives and add to total
        for v in range(len(vars_fisher)):
            vars_fisher[v] += np.square(ders[v])
            
    # divide totals by number of samples
    for v in range(len(vars_fisher)):
        vars_fisher[v] /= num_samples
    return vars_fisher