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

    A model is mades up of Components and Operators. Components are modified,
    usually (always?) by convolution, by Operators.
    * Don't like the work operator, think of something better
    """ 

    def __init__(self,pars):
        # here the pars apply to different components, need to have a system
        # of bookkeeping thats clear. (Maybe dict not best structure)
        self.pars = pars

    def __call__(self,times, freqs):
        """ Just feed times and freqs directly to components? """

        return self.func(times,freqs)


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
        model.func = lambda times, freqs: self.func(times, freqs) + \
                                          other.func(times, freqs)
        return model

    def __mul__(self,other):

        pars = {"comp1": self.pars, "comp2": other.pars}
        model = Model(pars)
        model.func = lambda times, freqs: self.func(times, freqs) * \
                                          other.func(times, freqs)
        return model

class Component(Model):
    """
    A model that is a function of either time or frequency. Not both.
    """
    def __init__(self,pars,dependant):

        super(Component,self).__init(pars)
        self.dependant = dependant

class GaussComp(Component):
    """
    A Gaussian.
    """
    
    def func(self,times,freqs):

        if self.dependant == "freq":
            x = freqs
            y = times
        elif self.dependant == "time":
            x = times
            y = freqs

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


class DeltaDM(Transform):
    
    def apply(self,old_func,times,freqs):
        delay = DM/(0.000241*freqs*freqs)
        new_func = old_func(times,freqs) # do something to old func
        return new_func

class ScatterTransform(Transform):
    """
    Convolve an arbitrary model with an exponential scattering function
    """

    def kernel(self,freqs,times):
    
       t0 = self.pars["t0"]
       tau_0 = self.pars["tau0"]
       f0 = self.pars["f0"]

       dt = times[1] - times[0]
       t = np.arange(0,10*tau_0,dt) 
       tau = tau_0 * (freqs / f0)**-4.
       scatter = np.exp( -1.*(t)[:,np.newaxis]/tau )
       scatter = np.append(np.zeros(scatter.shape),scatter,axis=0)
       print scatter.shape

       return scatter

    def kernel1d(self,freq,times):
    
       t0 = self.pars["t0"]
       tau_0 = self.pars["tau0"]
       f0 = self.pars["f0"]

       dt = times[1] - times[0]
       t = np.arange(0,100*tau_0,dt) # how long to make kernel? 
       tau = tau_0 * (freq / f0)**-4.
       scatter = np.exp( -1.*(t)/tau )
       scatter = np.append(np.zeros(scatter.shape),scatter,axis=0)

       return scatter

    def convolve(self,model):
        """
        Convolve the scattering function with the model.
        """

        pars = {"scatt": self.pars, "comp": model.pars}                                            
        new_model = Model(pars)                                                                         
        # convolution should be just in time with freq dependant tau, not 2d
        new_model.func = lambda times, freqs: scipy.signal.convolve2d(model(times,freqs),
                                                                  self.kernel(freqs,times).T,mode="same")
        #new_model.func = lambda times, freqs: self.func(times,freqs).T

        return new_model


    def convolve_singlefreq(self,model,times,freq):
        kernel = self.kernel1d(freq,times)
        at_freq = scipy.signal.fftconvolve(model(times,freq)[0], 
                                           kernel,mode="same") / np.sum(kernel)

        return at_freq

    def convolve_multi_freq(self,model,times,freqs):

        output = np.empty((len(freqs), len(times)))
        for i,freq in enumerate(freqs):
            output[i] = self.convolve_singlefreq(model,times,np.array([freq]))

        return output

    def __call__(self,model):
        pars = {"scatt": self.pars, "comp": model.pars}
        new_model = Model(pars) 
        new_model.func = lambda times, freqs: self.convolve_multi_freq(model,times,freqs)
        return new_model

