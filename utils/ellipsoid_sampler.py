# Describes code for weight space exploration for convex models  
import numpy as np
import tensorflow as tf
from keras import backend as K

class EllipsoidSampler():
    """
        Sample weight space with ellipsoid, centered at the optimal model ```\\theta``` 
    """

    def __init__(self, model, Xval, yval, FIM, exp):
        """ 
        Keyword arguments: 
        model: starting model 
        Xval: validation data 
        yval: validation labels 
        FIM: fisher information matrix 

        """
        self.model = model 
        self.orig_weights = model.get_weights()
        self.Xval = Xval
        self.yval = yval 
        self.fisher = FIM
        self.epsilon = exp['final_epsilon']
        self.delta = exp['delta']
        self.max_radius = exp['max_radius']

        # In order to sample from the surface of an ellipse, we use
        # https://math.stackexchange.com/questions/973101/how-to-generate-points-uniformly-distributed-on-the-surface-of-an-ellipsoid
        
        self.sigma = []
        nmax = 0.0
        for i in range(len(self.fisher)):
            if exp['ellipse_type'] == 'fisher':
                p_sigma = 1.0/(np.abs(self.fisher[i]) + 1e-5)
            else:
                p_sigma = 1.0/(np.abs(self.orig_weights[i]) + 1e-5)
            nmax = max(np.max(p_sigma), nmax)
            self.sigma.append(p_sigma)
        
        for i in range(len(self.sigma)):
            self.sigma[i] = self.sigma[i] / nmax
            self.sigma[i] = np.maximum(0.1, self.sigma[i])
        
    def evaluate_model(self, new_weights):
        """ Calculate the loss/accuracy of a new set of weights 
        Keyword arguments: 
        new_weights: list of var/weights to test
        
        Returns: accuracy of new_weights on (self.Xval, self.yval)
        """
        self.model.set_weights(new_weights)
        loss, acc = self.model.evaluate(self.Xval, self.yval, verbose = 0)
        return acc
    
    def restore_model(self):
        """
            Restore the model's weights to the original optimal. 
        """
        self.model.set_weights(self.orig_weights)

    def sample_ellipse(self, mvars, radius):
        """ 
        Uniformly sample a point from the surface of an ellipsoid

        Keyword arguments
        mvars: list of layer weights [(w1, b1), (w2, b2)...] corresponding to sphere 
        center 
        radius: radius of sphere

        Returns: 
        point: a sampled point on the surface of sphere 
        """

        point = []
        sum_sq = 0
        for i in range(len(mvars)):
            axes = self.sigma[i]*radius
            gauss = np.random.normal(0.0, axes, mvars[i].shape) # sample from standard normal gaussian 
            point.append(gauss)
            sum_sq += np.sum(np.square(gauss) / np.square(axes))
        norm = np.sqrt(sum_sq)
        for i in range(len(point)):
            point[i] = point[i] / norm 
            point[i] = point[i]+ mvars[i]
        
        
        return point
    
    def get_distance(self, vars1, vars2):
        assert(len(vars1) == len(vars2))
        distance = 0
        for i in range(len(vars1)):
            distance += np.sum(np.square(vars1[i] - vars2[i]))
        
        return np.sqrt(distance)

    def sample_weight_space(self, num_samples, max_radius, verbose = False):
        """
        Build an epsilon space for specified epsilon, performing binary search over radii
       
        Keyword arguments 
        epsilon: cutoff accuracy 
        num_samples: number of models to sample at each radius 
        prop_weights: proportion of top fisher weights to explore over 
        radius_inc: amount to increment radius at every iteration  

        Returns: nothing, but stores the following internal values: 
        self.ep_radius: radius for resulting epsilon space 
        """
        tol = 0.1
        radius = self.delta
        upper_lim = max_radius 
        lower_lim = 0.0
        while True:
            radius = (upper_lim + lower_lim)/2.0 
            successful = True
            for i in range(num_samples):
                pert_weights = self.sample_ellipse(self.orig_weights, radius)
                acc = self.evaluate_model(pert_weights)
                if verbose:
                    dist = self.get_distance(pert_weights, self.orig_weights)
                    print("Iter-{:d}. Radius: {:0.02f}. Accuracy: {:0.003f}. Distance: {:0.02f}".format(i, radius, acc, dist))
                if acc < self.epsilon:
                    successful = False
                    break 
            
            # update radius limits
            if successful: 
                # increase lower limit 
                lower_lim = (upper_lim + lower_lim)/2.0
            else:
                # decrease upper limit 
                upper_lim = (upper_lim + lower_lim)/2.0 
            # check tolerance 
            if upper_lim - lower_lim < tol:
                self.restore_model()
                self.ep_radius = (upper_lim + lower_lim) / 2.0 
                return
            
        
    
def get_mask_weight_distance(weights1, mask1, mask2, weights2):
    distance = 0.0
    assert len(weights1) == len(weights2)
    for i in range(len(weights1)):
        distance  += np.sum(np.square(np.multiply(mask1[i], weights1[i]) - np.multiply(mask2[i], weights2[i])))
    return np.sqrt(distance)
        
def ellipsoid_intersection(epsilon_spaces):

    # initialize with weight average 
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
        r = ep_space.ep_radius
        sigma = ep_space.sigma
        num_vars = len(agent_vars)

        tf_sigma = []
        tf_agent_vars = []
        # create tensorflow arrays
        for v in range(num_vars):
            tf_agent_vars.append(tf.constant(agent_vars[v], dtype=tf.float32))
            tf_sigma.append(tf.constant(sigma[v]*r, dtype=tf.float32)) 

        dist = tf.reduce_sum(tf.square(tf_agent_vars[0] - gems_vars[0]) / tf.square(tf_sigma[0]))
        dist = dist + tf.reduce_sum(tf.square(tf_agent_vars[1] - gems_vars[1]) / tf.square(tf_sigma[1]))
        dist = tf.sqrt(dist)
        obj += tf.maximum(0.0, dist - 1.0)

    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train_op = optimizer.minimize(obj)

    num_steps = 100000
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(gems_vars[0].assign(weight_avg[0]))
    sess.run(gems_vars[1].assign(weight_avg[1]))
    for i in range(num_steps):
        _, obj_i = sess.run([train_op, obj])
        if obj_i == 0.0:
            break
    grad = sess.run(tf.gradients(obj, gems_vars))
    

    gvars = sess.run(gems_vars)
    # Check to see if it's in the intersection 
    for t in range(len(epsilon_spaces)):
        psum = 0.0
        for i in range(2):
            sigma = epsilon_spaces[t].sigma[i]*epsilon_spaces[t].ep_radius
            psum = psum + np.sum(np.square(gvars[i] - epsilon_spaces[t].orig_weights[i]) / np.square(sigma))



    return sess.run(gems_vars), True # TODO: FIX THIS
    
    #gems_model = create_logit(intersected_weights, Xts_flat, yts_flat, bias=True)
    #loss, acc = gems_model.evaluate(Xts_flat, yts_flat, verbose=0)
    #print("Epsilon: {:0.2f}. Accuracy: {:0.03f} Loss: {:0.02f}".format(ep, acc, loss))


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
    fisher_vars1 = e1.adaptive_radius
    fisher_vars2 = e2.adaptive_radius
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