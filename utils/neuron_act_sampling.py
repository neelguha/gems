# Contains code for performing GEMS on individual neurons 

import keras.backend as K
import numpy as np

class NeuronActSampler():
    """
        Concentric sphere sampler, uses the neuron activation to derive radius
    """

    def __init__(self, model, Xval, yval, FIM, layer_index, neuron_index, slack, exp):
        """ 
        Represent a neuron as (w, b).
        Keyword arguments: 
            model: starting model 
            Xval: validation data 
            yval: validation labels 
            FIM: fisher information matrix 
            layer_index: the index layer for the model 
            neuron_index: the index of the neuron
        """

        self.Xval = Xval
        self.yval = yval
        self.num_samples = len(Xval)

        self.model = model 
        self.full_weights = model.get_weights()
        self.layer_weights = model.layers[layer_index].get_weights() 
        self.orig_weights = [self.layer_weights[0][:, neuron_index], self.layer_weights[1][neuron_index]] # [w, b]

        self.layer_index = layer_index
        self.neuron_index = neuron_index
        self.full_fisher = FIM
        self.layer_fisher = [self.full_fisher[layer_index], self.full_fisher[layer_index+1]]
        self.prop_weights = exp['fisher_prop']
        self.slack = slack
        
        self.z = self.get_z()
        self.mask = self.get_mask(self.prop_weights, neuron_index)

    def get_z(self):
        layer_output = self.model.layers[0].output
        input = self.model.input
        func = K.function([input], [layer_output])
        activations = func([self.Xval])[0]
        return activations[:, self.neuron_index]


    def get_distance(self, vars1, vars2):
        assert(len(vars1) == len(vars2))
        distance = 0
        for i in range(len(vars1)):
            distance += np.sum(np.square(vars1[i] - vars2[i]))
        
        return np.sqrt(distance)
    
    def get_mask(self, prop, neuron_index):
        """
        Returns a mask, which is applied when exploring the weight space (i.e. restricting 
        our attention to only the important dimensions). A mask is applied if the fisher info
        for a parameter is less than the defined threshold (unimporant), or if the weights do 
        not belong to the specified neuron. 

        Keyword arguments: 
            prop: proportion of layer weights to perform search over (i.e. top 10%)
            neuron_index: index of neuron to be perturbed
        Returns:
            mask: shape of layer weights
        """

        # Determine mask for unimportant variables 
        layer_f = np.concatenate([v.ravel() for v in self.layer_fisher])
        cutoff = np.percentile(layer_f, (1-prop)*100)
        mask = []
        
        # weight
        fisher_w = self.layer_fisher[0] >= cutoff
        mask.append(fisher_w[:, self.neuron_index])

        # bias
        fisher_b = self.layer_fisher[1] >= cutoff
        mask.append(fisher_b[self.neuron_index])
        
        return mask
        
    def evaluate_model(self, new_weights):
        """ Calculate the loss/accuracy of a new set of weights 
        Keyword arguments: 
        new_weights: list of var/weights to test
        
        Returns: accuracy of new_weights on (self.Xval, self.yval)
        """
        z_prime = np.maximum(np.dot(self.Xval, new_weights[0]) + new_weights[1], 0)
        return np.linalg.norm(self.z-z_prime) / self.num_samples

    def restore_model(self):
        """
            Restore the model's weights to the original optimal. 
        """
        self.model.set_weights(self.full_weights)

class ConcenActNeuronSphereSampler(NeuronActSampler):
    """
        Sample weight space with concentric spheres for a single neuron
    """

    def __init__(self, model, Xval, yval, FIM, layer_index, neuron_index, slack, exp):
        """
            Keyword arguments: 
                radius_inc: how much to increase the radius on each iteration
        """
        NeuronActSampler.__init__(self, model, Xval, yval, FIM, layer_index, neuron_index, slack, exp)
        self.radius_inc = exp['delta']
        self.max_radius = exp['max_radius']

        
    def sample_uniform(self, radius):
        """ 
        Uniformly sample a point from the surface of a sphere

        Keyword arguments
            radius: radius of sphere

        Returns: 
            point: a sampled point on the surface of sphere 
        """

        point = []
        sum_sq = 0
        for i in range(len(self.orig_weights)):
            gauss = np.random.normal(0.0, 1.0, self.orig_weights[i].shape) # sample from standard normal gaussian 
            gauss = np.multiply(gauss, self.mask[i]) # mask non-searched weights
            point.append(gauss)
            sum_sq += np.sum(np.square(gauss))
        norm = np.sqrt(sum_sq)

        for i in range(len(point)):
            point[i] = radius*point[i] / norm + self.orig_weights[i]
        
        return point


    def sample_weight_space(self, num_samples, verbose = False):
        """
        Build an epsilon space for specified epsilon, performing binary search over radii
       
        Keyword arguments 
        epsilon: cutoff accuracy 
        num_samples: number of models to sample at each radius 
        radius_inc: amount to increment radius at every iteration  

        Returns: nothing, but stores the following internal values: 
        self.ep_radius: radius for resulting epsilon space 
        """
        
        #tol = 0.01
        tol = 1.0
        radius = self.radius_inc # initialize radius from delta
        upper_lim = self.max_radius
        lower_lim = 0.0
        while True:    
            radius = (upper_lim + lower_lim) / 2.0
            successful = True
            for i in range(num_samples):
                pert_weights = self.sample_uniform(radius)
                dist = self.evaluate_model(pert_weights)
                if verbose:
                    print("Iter-{:d}. Radius: {:0.02f}. Dist: {:0.003f}".format(i, radius, dist))
                if dist > self.slack:
                    if verbose:
                        print("Performance degraded. Ending.")
                    successful = False
                    break
            
            # update radius limits
            if successful: 
                # increase lower limit 
                lower_lim = (upper_lim + lower_lim)/2.0
            else:
                # decrease upper limit 
                upper_lim = (upper_lim + lower_lim)/2.0
            
            if verbose:
                print("Radius = %f" % radius)   
            
            # check tolerance 
            if upper_lim - lower_lim < tol:
                self.restore_model()
                self.ep_radius = (upper_lim + lower_lim) / 2.0 
                return


class ConcenNeuronActIntersector():
    """
        Calculates intersections for neuron activations
    """

    def get_distance(self, ep1, ep2):        
        dist = 0
        for i in range(len(ep1.orig_weights)):
            mask1, mask2 = ep1.mask[i], ep2.mask[i]
            param1 = np.multiply(mask1, ep1.orig_weights[i])
            param2 = np.multiply(mask2, ep2.orig_weights[i])
            dist += np.sum(np.square(param1 - param2))
        dist =  np.sqrt(dist)
        return dist

    def check_intersection(self, ep1, ep2):
        dist = self.get_distance(ep1, ep2)
        return dist <= ep1.ep_radius + ep2.ep_radius


    def neuron_act_intersection(self, e1, e2, n1, n2 ):
        """ Return model in the intersection of passed epsilon space. Currently only works for two agents. 
        For any pair of parameters, there are three cases: 

            1) both parameters are unimportant (fim = 0): average the parameters 
            2) one parameter is important (fim > 0) and one is unimportant (fim = 0): set to important parameter val 
            3) both parameters are important: take weighted average according to radii 

        Keyword arguments:
        e1: SingleSphere epsilon space for first agent. 
        n1: neuron index for agent 2
        e2: SingleSphere epsilon space for second agent 
        n2: neuron index for agent 2

        Returns weights for intersected model if it exists, None else.
        """

        num_vars = len(e1.orig_weights)
        intersection_vars = []
        r1 = e1.ep_radius
        r2 = e2.ep_radius

        # Intersect weights
        mask1 = e1.mask[0].astype(float)
        mask2 = e2.mask[0].astype(float)
        z1 = np.zeros_like(mask1) # multipliers for model 1's weights
        z2 = np.zeros_like(mask2) # multipliers for model 2's weights 

        # When both parameters are important 
        z1 += r2 / (r1 + r2)*np.multiply(mask1, mask2)
        z2 += r1 / (r1 + r2)*np.multiply(mask1, mask2)

        # When both parameters are unimportant
        z1 += 0.5*(np.multiply(1.0 - mask1, 1.0 - mask2))
        z2 += 0.5*(np.multiply(1.0 - mask1, 1.0 - mask2))

        # When one set of parameters is important
        z1 += 1.0*(np.multiply(mask1, 1.0 - mask2))
        z2 += 1.0*(np.multiply(1.0 - mask1, mask2))

        w = z1*e1.orig_weights[0] + z2*e2.orig_weights[0]
        
        # Intersect bias 
        mask1 = e1.mask[1].astype(float)
        mask2 = e2.mask[1].astype(float)
        b1 = e1.orig_weights[1]
        b2 = e2.orig_weights[1]
        if mask1 == 0 and mask2 == 0:
            b = np.mean(b1, b2)
        elif mask1 == 1 and mask2 == 1:
            z1 = r2 / (r1 + r2)
            z2 = r1 / (r1 + r2)
            b = z1*b1 + z2*b2
        else: 
            b = mask1*b1 + mask2*b2
        
        return w, b