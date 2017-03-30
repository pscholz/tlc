import numpy as np
import scipy.stats
import scipy.signal
import matplotlib.pyplot as plt

class Model(object):
    """
    A structure that provides the intensity as a function of time and frequency
    given a functional form defined by the model and a set of parameters for
    the model.
    
    Parameters
    ----------
    
    pars : dict
        A dictionary of all the free parameters of the model.

    Notes
    -----

    A Model is mades up of Components. Components have only functional form
    in one dimension.

    Models are modified by Transforms.
    """ 

    def __init__(self,pars):
        # here the pars apply to different components, need to have a system
        # of bookkeeping thats clear. (Maybe dict not best structure)
        self.pars = pars

    def __call__(self,times, freqs):
        """ Just feed times and freqs directly to components? """

        return self.function(times,freqs)


    def get_intensity_values(times, freqs):
        """
        For a set of parameters, return an array of intensity values for
        a given set of times and freqs.

        Parameters
        ----------
   
        times 


        Notes
        -----

        Could have times and freqs set as attributes of the class instead
        of fed to this function. 
        * But, I don't like this: time and freq span and resolution are 
          properties of the data, not model.
        """
        pass

    def plot(self, times, freqs, ax=None):
        """
        Plot the form of the model in time and frequency.
        """ 

        if ax is None:
            ax_im = plt.axes((0.15, 0.15, 0.6, 0.6))
            ax_ts = plt.axes((0.15, 0.75, 0.6, 0.2),sharex=ax_im)
            ax_spec = plt.axes((0.75, 0.15, 0.2, 0.6),sharey=ax_im)

        ax_im.pcolormesh(times,freqs,self(times,freqs),cmap="YlOrBr_r")
        ax_ts.plot(times,self(times,freqs).sum(0),c="orangered")
        ax_spec.plot(self(times,freqs).sum(1),freqs,c="orangered")
        ax_im.set_xlabel("Time")
        ax_im.set_ylabel("Frequency")

    def __add__(self,other):

        pars = {"comp1": self.pars, "comp2": other.pars}
        model = Model(pars)
        model.function = lambda times, freqs: self.function(times, freqs) + \
                                              other.function(times, freqs)
        return model

    def __mul__(self,other):

        pars = {"comp1": self.pars, "comp2": other.pars}
        model = Model(pars)
        model.function = lambda times, freqs: self.function(times, freqs) * \
                                              other.function(times, freqs)
        return model

class Component(Model):
    """
    A Model that is a function of either time or frequency. Not both.
    """
    def __init__(self,pars,dependant):
        super(Component,self).__init__(pars)
        self.dependant = dependant

class GaussComp(Component):
    """
    A Gaussian.
    """
    
    def function(self,times,freqs):

        if self.dependant == "freq":
            x = freqs
            y = times
        elif self.dependant == "time":
            x = times
            y = freqs

        x = np.atleast_1d(x)
        y = np.atleast_1d(y)

        pars = self.pars
        gauss = scipy.stats.norm(pars["mean"],pars["width"])

        x_and_y = np.tile(pars["norm"]*gauss.pdf(x),len(y))
        x_and_y.shape = (len(y),len(x))

        if self.dependant == "freq":
            x_and_y = x_and_y.T

        return x_and_y

class Transform(object):
    """ A Transform modifies a Model in some way.
    """
    def __init__(self,pars):
        self.pars = pars


    def __call__(self,model):
        pars = {self.name: self.pars, "comp": model.pars}
        new_model = Model(pars) 
        new_model.function = lambda times, freqs: self.function(model,times,freqs)
        return new_model


class DeltaDMTransform(Transform):
    """
    Shift times in each frequency channel by a dispersion delay.
    As the input is expected to be dedispersed to some DM already, this should
    be a change in DM from that fiducial value.
    """

    def __init__(self,pars):
        super(DeltaDMTransform,self).__init__(pars)
        self.name = "deltadm"
    
    def function(self,model,times,freqs):
        freqs = np.atleast_1d(freqs)
        times = np.atleast_1d(times)

        dt = times[1] - times[0]
        delays = self.pars["dm"]/(0.000241*freqs*freqs)

        output = np.empty((len(freqs), len(times)))
        for i,freq in enumerate(freqs):
            output[i] = model(times-delays[i],freq)

        return output


class ScatterTransform(Transform):
    """
    Convolve an arbitrary model with an exponential scattering function
    """

    def __init__(self,pars):
        super(ScatterTransform,self).__init__(pars)
        self.name = "scatter"

    def _kernel(self,freq,times):
    
       tau_0 = self.pars["tau0"]
       f0 = self.pars["f0"]

       dt = times[1] - times[0]
       t = np.arange(0,100*tau_0,dt) # how long to make kernel? 
       tau = tau_0 * (freq / f0)**-4.
       scatter = np.exp( -1.*(t)/tau )
       scatter = np.append(np.zeros(scatter.shape),scatter,axis=0)

       return scatter

    def _convolve_singlefreq(self,model,times,freq):
        kernel = self._kernel(freq,times)
        at_freq = scipy.signal.fftconvolve(model(times,freq)[0], 
                                           kernel,mode="same") / np.sum(kernel)

        return at_freq

    def function(self,model,times,freqs):

        freqs = np.atleast_1d(freqs)
        times = np.atleast_1d(times)
        output = np.empty((len(freqs), len(times)))
        for i,freq in enumerate(freqs):
            output[i] = self._convolve_singlefreq(model,times,freq)

        return output
