"""
Provides a framework for defining 1D and 2D models to fit dynamic spectra.
"""

import numpy as np
import scipy.stats
import scipy.signal
import matplotlib.pyplot as plt

class Parameter(object):
    """
    An object that describes a parameter in a model.

    Parameters
    ----------

    name : str
        Name of the parameter.

    value : float
        Value of the parameter.

    """

    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.fixed = True
        self.free = (not self.fixed)
        self.model = None
        self.piece = None

    def set_fixed(self,fixed=True):
        self.fixed = fixed
        self.free = (not fixed)

    def set_free(self,free=True):
        self.fixed = (not free)
        self.free = free

    def get_component(self):
        return self.component

    def __str__(self):
        return "Parameter: " + str(self.name) + " Value: " + str(self.value) +\
                " Component: " + repr(self.piece)

    def __repr__(self):
        return str(self.name) + ": " + str(self.value)

class ParList(object):
    """
    A list of Parameter objects.

    Parameters
    ----------

    names : list
        List of names for each parameter in `pars`.

    pars : list
        List of Parameter objects.

    """

    def __init__(self,names,pars):
        self._names = list(names)
        self._pars = list(pars)

    def __setitem__(self,item,value):
        if type(item) is str:
            self._pars[self._names.index(item)].value = value
        elif type(item) is int:
            self._pars[item].value = value
        else:
            raise KeyError("%s of type %s is not a valid index" % (item,repr(type(item))))

    def __getitem__(self,item):
        if type(item) is str:

            return self._pars[self._names.index(item)]
        elif type(item) is int:
            return self._pars[item]

        else:
            raise KeyError("%s of type %s is not a valid index" % (item,repr(type(item))))

    def __iter__(self):
        return iter(self._pars)

    def __repr__(self):
        return repr(self._pars)

    @property
    def names(self):
        return self._names

    @property
    def pars(self):
        return self._pars


class Model(object):
    """
    A structure that provides the intensity as a function of time and frequency
    given a functional form defined by the model and a set of parameters for
    the model.

    Parameters
    ----------

    pars : ParList
        A ParList object of all the parameters of the model.

    Notes
    -----

    A Model is mades up of Components. Components have only functional form
    in one dimension.

    Models are modified by Transforms.
    """

    def __init__(self,pars=None):
        # here the pars apply to different components, need to have a system
        # of bookkeeping thats clear. (Maybe dict not best structure)

        if pars is None:
            pars = self.get_blank_params()

        self.pars = pars

    def __call__(self,times, freqs):
        """ Just feed times and freqs directly to components? """

        return self.function(times,freqs)


    def get_intensity_values(self, times, freqs):
        """
        For a set of parameters, return an array of intensity values for
        a given set of times and freqs.

        Parameters
        ----------

        times : ndarray
            Array of times for which to calculate intensity values.

        freqs : ndarray
            Array of frequencies for which to calculate intensity values.

        Returns
        -------

        ndarray
            2D array of intensity values (in S/N).

        Notes
        -----

        Could have times and freqs set as attributes of the class instead
        of fed to this function.
        * But, I don't like this: time and freq span and resolution are
          properties of the data, not model.
        """

        return self(times,freqs)

    def set_pars(self,value_array):
        """
        Given an array of parameter values, set parameters for the model.

        Note: this requires knowledge of how parameters are ordered in dict...
              i.e. iffy...
        """
        i = 0
        for par in self.pars:

            if par.free:
                par.value = value_array[i]
                i += 1

    def plot(self, times, freqs, ax=None):
        """
        Plot the form of the model in time and frequency.
        """

        if ax is None:
            ax_im = plt.axes((0.15, 0.15, 0.6, 0.6))
            ax_ts = plt.axes((0.15, 0.75, 0.6, 0.2),sharex=ax_im)
            ax_spec = plt.axes((0.75, 0.15, 0.2, 0.6),sharey=ax_im)

        #ax_im.pcolormesh(times,freqs,self(times,freqs),cmap="YlOrBr_r")
        ax_im.pcolormesh(times,freqs,self(times,freqs),cmap="plasma")
        ax_ts.plot(times,self(times,freqs).sum(0),c="C1")
        ax_spec.plot(self(times,freqs).sum(1),freqs,c="C1")
        ax_im.set_xlabel("Time")
        ax_im.set_ylabel("Frequency")

    @staticmethod
    def new_composite_model(*args):
        """
        Create a new ParList object for that consists of the parameters from
        each Model in `args`. Initiate a new Model with that ParList.

        Parameters
        ----------

        *args : Model
            Each argument should be a Model object that will make up the total
            model.

        Returns
        -------

        model : Model
            The newly initiated Model created from the merged ParList.

        Notes
        -----

        Parameter names are appended with an index for each "piece" of the
        model (a piece is a model Component or Transform). So, for example,
        in a model that is a 2D Gaussian, the parameters become `mean0`,
        `width0`, `norm0`, `mean1`, `width1`, `norm1`.

        This method does _not_ attach a `function` to the model, so does not
        define how the model pieces are related. That has to be done
        separately.

        """

        pieces = []
        new_par_keys = []
        new_par_objs = []

        for model in args:

            pieces += model.pieces
            for par in model.pars:
                par.name = par.name + str(pieces.index(par.piece))
                new_par_keys += [par.name]
            new_par_objs += model.pars

        new_model = Model(ParList(new_par_keys, new_par_objs))
        new_model.pieces = pieces

        return new_model

    def __add__(self,other):

        model = Model.new_composite_model(self,other)

        model.function = lambda times, freqs: self.function(times, freqs) + \
                                              other.function(times, freqs)
        return model

    def __mul__(self,other):

        model = Model.new_composite_model(self,other)

        model.function = lambda times, freqs: self.function(times, freqs) * \
                                              other.function(times, freqs)
        return model

class Component(Model):
    """
    A Model that is a function of either time or frequency.

    Optional Parameters
    -------------------

    pars : ParList
        A ParList object that contains the parameters for the model.
        Default is `None` which will use `get_blank_params` to create a
        new ParList with default values for the parameters.

    dependant : string
        The dependant coorinate for the component. Either "time" or "freq".
        Default is "time".

    """

    _par_names = []
    _par_dummy_vals = []

    def __init__(self, pars=None, dependant="time"):
        super(Component,self).__init__(pars)
        self.dependant = dependant

        for par in self.pars:
            par.piece = self

        self.pieces = [self]

        self.name = self._short_name + " " + self.dependant

    def function(self,times,freqs):
        """
        Take the 1D function for the component and tile it so that it is
        a function of both time and freq.

        Parameter
        ---------

        times : array_like
            Array of times to calculate intensity signal for.

        freqs : array_like
            Array of frequencies to calculate intensity signal for.

        """

        if self.dependant == "freq":
            x = freqs
            y = times
        elif self.dependant == "time":
            x = times
            y = freqs

        x = np.atleast_1d(x)
        y = np.atleast_1d(y)

        x_and_y = np.tile(self.function_1d(x),len(y))
        x_and_y.shape = (len(y),len(x))

        if self.dependant == "freq":
            x_and_y = x_and_y.T

        return x_and_y

    def function_1d(self,x):
        """
        Given an array of dependant coordinates (times _or_ frequencies)
        return the value of the model for those coordinates.

        Parameters
        ----------

        x : array_like
            Array of dependant coordinate for the Component. Either times or
            frequencies.

        Returns
        -------

        ndarray
            Array of model intensities.

        """
        raise ImplementationError("function_1d needs to be implemented in "\
                                  "subclass.")

    @classmethod
    def get_blank_params(cls):
        """
        Initiate a ParList object with default values for the model.

        Returns
        -------

        ParList
            A list of Parameters set to their default values.

        """

        par_objs = []
        for name, val in zip(cls._par_names, cls._par_dummy_vals):
            par_objs.append(Parameter(name,val))

        return ParList(cls._par_names, par_objs)

class GaussComp(Component):
    """
    A Gaussian model component.
    """

    _par_names = ["norm","mean","width"]
    _par_dummy_vals = [1.0,1.0,1.0]
    _short_name = "gauss"

    def function_1d(self,x):

        pars = self.pars
        gauss = scipy.stats.norm(pars["mean"].value,pars["width"].value)

        return pars["norm"].value*gauss.pdf(x)

class PowerLawComp(Component):
    """
    A Power Law model component.

    """

    _par_names = ["norm","index"]
    _par_dummy_vals = [1.0,0.0]
    _short_name = "powlaw"

    def function_1d(self,x):

        pars = self.pars

        return pars['norm'].value*(x/x[0])**pars["index"].value

class Transform(object):
    """ A Transform modifies a Model in some way.
    """
    def __init__(self,pars):
        self.pars = pars

        for par in self.pars:
            par.piece = self

        self.pieces = [self]

    def __call__(self,model):

        new_model = Model.new_composite_model(self,model)

        new_model.function = lambda times, freqs: self.function(model,times,freqs)

        return new_model

    @classmethod
    def get_blank_params(cls):

        pars = []
        for name, val in zip(cls._par_names, cls._par_dummy_vals):
            pars.append(Parameter(name,val))

        return dict(zip(cls._par_names, pars))


class DeltaDMTransform(Transform):
    """
    Shift times in each frequency channel by a dispersion delay.
    As the input is expected to be dedispersed to some DM already, this should
    be a change in DM from that fiducial value.
    """

    _par_names = ["dm"]
    _par_dummy_vals = [1.0]

    def __init__(self,pars):
        super(DeltaDMTransform,self).__init__(pars)
        self.name = "deltadm"

    def function(self,model,times,freqs):
        freqs = np.atleast_1d(freqs)
        times = np.atleast_1d(times)

        dt = times[1] - times[0]
        delays = self.pars["dm"].value/(0.000241*freqs*freqs)

        output = np.empty((len(freqs), len(times)))
        for i,freq in enumerate(freqs):
            output[i] = model(times-delays[i],freq)

        return output


class ScatterTransform(Transform):
    """
    Convolve an arbitrary model with an exponential scattering function
    """

    _par_names = ["tau0","f0"]
    _par_dummy_vals = [1.0,1.0]

    def __init__(self,pars):
        super(ScatterTransform,self).__init__(pars)
        self.name = "scatter"

    def _kernel(self,freq,times):

       tau_0 = self.pars["tau0"].value
       f0 = self.pars["f0"].value

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
