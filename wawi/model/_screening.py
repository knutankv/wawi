import numpy as np
import json
from ._hydro import Seastate
from pathlib import Path

'''
SCREENING SUBMODULE
'''

class ScreeningCase:
    """
    Represents a screening case for analyzing the behavior of a system under different seastates and parameter variations.

    Attributes
    ----------
    name : str, optional
        Name of the screening case. Defaults to None.
    seastate : Seastate
        The seastate object representing the environmental conditions.
    parvar : dict
        A dictionary defining the parameter variations. Keys are parameter names, and values are arrays of parameter values.
    independent : bool, optional
        If True, the parameter variations are treated as independent, and all combinations are generated.
        If False, the parameter variations are treated as dependent, and the arrays must have the same length. Defaults to True.
    combos : list of dict
        A list of dictionaries, where each dictionary represents a combination of parameter values.
    ix : int
        The index of the current combination.

    Methods
    -------
    assign_parvar(parvar)
        Assigns and processes the parameter variations.
    get_parameter_space()
        Generates the parameter space based on the parameter variations and the 'independent' flag.
    from_json(json_file, **kwargs)
        Creates a ScreeningCase object from a JSON file.
    get_combo()
        Returns the current combination of parameter values.
    get_next_combo()
        Iterates to the next combination and returns it.
    iterate_seastate()
        Updates the seastate object with the next combination of parameter values.
    get_next_seastate()
        Iterates to the next seastate and returns it.
    iterate_ix()
        Iterates the index to the next combination.
    reset_ix()
        Resets the index to the first combination.

    Notes
    -----
    This documentation was automatically generated using GitHub Copilot in conjunction with the Gemini language model.
    """
    def __init__(self, seastate, parvar, independent=True, name=None):
        """
        Initializes a ScreeningCase object.

        Parameters
        ----------
        seastate : Seastate
            The seastate object representing the environmental conditions.
        parvar : dict
            A dictionary defining the parameter variations. Keys are parameter names, and values are arrays of parameter values.
        independent : bool, optional
            If True, the parameter variations are treated as independent, and all combinations are generated.
            If False, the parameter variations are treated as dependent, and the arrays must have the same length. Defaults to True.
        name : str, optional
            Name of the screening case. Defaults to None.

        Raises
        ------
        ValueError
            If dependent parameter arrays are requested, they must have the same length.
        """
        self.name = name
        self.seastate = seastate
        self.assign_parvar(parvar)
        self.independent = independent
        self.combos = self.get_parameter_space()
        self.ix = -1

        if not self.independent:
            sz_prev = None
            for key in self.parvar:
                sz = len(self.parvar[key])
                if sz_prev is not None and sz != sz_prev:
                    raise ValueError('If dependent parameter arrays are requested, they must have the same length!')
                sz_prev = sz * 1

    def assign_parvar(self, parvar):
        """
        Assigns and processes the parameter variations.

        Parameters
        ----------
        parvar : dict
            A dictionary defining the parameter variations. Keys are parameter names, and values are arrays of parameter values.

        Returns
        -------
        None
        """
        self.parvar = dict()
        for key in parvar:
            if type(parvar[key]) is str:
                self.parvar[key] = eval(parvar[key])
            else:
                self.parvar[key] = np.array(parvar[key])

        # Convert angles
        conversions = {'theta0': np.pi/180.0, 'thetaU': np.pi/180.0}
        for key in self.parvar:
            if key in conversions:
                self.parvar[key] = self.parvar[key] * conversions[key]

    def get_parameter_space(self):
        """
        Generates the parameter space based on the parameter variations and the 'independent' flag.

        Returns
        -------
        list of dict
            A list of dictionaries, where each dictionary represents a combination of parameter values.
        """
        pars = [self.parvar[k] for k in self.parvar]
        keys = [k for k in self.parvar if k]

        if self.independent:
            combos = np.array(np.meshgrid(*pars)).reshape(len(keys), -1).T
        else:
            combos = np.vstack(pars).T

        combo_dicts = [dict(zip(keys, combo)) for combo in combos]
        return combo_dicts

    @property
    def n(self):
        """
        Returns the number of combinations in the parameter space.

        Returns
        -------
        int
            The number of combinations.
        """
        if self.independent:
            return np.prod([len(v) for v in self.parvar.values()])
        else:
            return len(list(self.parvar.values())[0])

    @classmethod
    def from_json(cls, json_file, **kwargs):
        """
        Creates a ScreeningCase object from a JSON file.

        Parameters
        ----------
        json_file : str
            Path to the JSON file.
        **kwargs : dict
            Additional keyword arguments to be passed to the Seastate.from_json method.

        Returns
        -------
        ScreeningCase
            A ScreeningCase object.
        """
        with open(json_file, 'r') as fileobj:
            data = json.load(fileobj)

        seastate = Seastate.from_json(data['seastate'], **kwargs)

        # Update options if provided (to enable overriding options from screening setup)
        if 'options' in data:
            options = data['options']
        else:
            options = {}

        if 'pontoon_options' in data:
            pontoon_options = data['pontoon_options']
        else:
            pontoon_options = {}

        seastate.options.update(**options)
        seastate.pontoon_options.update(**pontoon_options)

        parvar = data['parvar']
        if 'independent' in data:
            independent = data['independent']
        else:
            independent = True

        return cls(seastate, parvar, independent=independent, name=Path(json_file).stem)

    def get_combo(self):
        """
        Returns the current combination of parameter values.

        Returns
        -------
        dict
            A dictionary representing the current combination of parameter values.
        """
        return self.combos[self.ix]

    def get_next_combo(self):
        """
        Iterates to the next combination and returns it.

        Returns
        -------
        dict
            A dictionary representing the next combination of parameter values.
        """
        self.iterate_ix()
        combo = self.combos[self.ix]
        return combo

    def iterate_seastate(self):
        """
        Updates the seastate object with the next combination of parameter values.

        Returns
        -------
        Seastate
            The updated seastate object.
        """
        combo = self.get_next_combo()
        if combo is not None:
            for key in combo:
                setattr(self.seastate, key, combo[key])
        return self.seastate

    def get_next_seastate(self):
        """
        Iterates to the next seastate and returns it.

        Returns
        -------
        Seastate
            The next seastate object.
        """
        self.iterate_seastate()
        return self.seastate

    def iterate_ix(self):
        """
        Iterates the index to the next combination.

        If the index reaches the end of the parameter space, it resets to the beginning.

        Returns
        -------
        None
        """
        if self.ix == (self.n - 1):
            self.ix = 0  # reset
        else:
            self.ix += 1

    def reset_ix(self):
        """
        Resets the index to the first combination.

        Returns
        -------
        None
        """
        self.ix = 0
