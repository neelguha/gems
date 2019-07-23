# Perform GEMS for convex models 
import numpy as np

class ConcenSphereSampler():
    """
        Sample weight space with concentric spheres, centered at the optimal model ```\\theta``` 
    """

    def __init__(self, model, Xval, yval, FIM, epsilon, exp):
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
        self.epsilon = epsilon
        self.prop_weights = exp['fisher_prop']
        self.delta = exp['delta']
        self.max_radius = exp['max_radius']
        
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

    def sample_uniform(self, mvars, mask, radius):
        """ 
        Uniformly sample a point from the surface of a sphere

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
            gauss = np.random.normal(0.0, 1.0, mvars[i].shape) # sample from standard normal gaussian 
            gauss = np.multiply(gauss, mask[i])
            point.append(gauss)
            sum_sq += np.sum(np.square(gauss))
        norm = np.sqrt(sum_sq)

        for i in range(len(point)):
            point[i] = radius*point[i] / norm + mvars[i]
        
        return point

    def get_top_mask(self, prop):
        """
        Keyword arguments: 
            prop: proportion of vars to perform search over (i.e. top 10%)
        """
        all_f = np.concatenate([v.ravel() for v in self.fisher])
        cutoff = np.percentile(all_f, (1-prop)*100)
        mask = [] # mask fisher variables
        for i in range(len(self.fisher)):
            mask.append(self.fisher[i] >= cutoff)
        return mask
    
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
        tol = 0.01
        self.mask = self.get_top_mask(self.prop_weights)
        radius = self.delta
        upper_lim = max_radius 
        lower_lim = 0.0
        while True:
            radius = (upper_lim + lower_lim)/2.0 
            successful = True
            for i in range(num_samples):
                pert_weights = self.sample_uniform(self.orig_weights, self.mask, radius)
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
        
