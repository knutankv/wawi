import numpy as np
import json
from ._hydro import Seastate
from pathlib import Path

'''
SCREENING SUBMODULE
'''

class ScreeningCase:
    def __init__(self, seastate, parvar, independent=True, name=None):
        
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
                
                if sz_prev is not None and sz!=sz_prev:
                    raise ValueError('If dependent parameter arrays are requested, they must have the same length!')
                    
                sz_prev = sz*1
        
        
    def assign_parvar(self, parvar):
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
                self.parvar[key] = self.parvar[key]*conversions[key]
                
            
    def get_parameter_space(self):
        pars = [self.parvar[k] for k in self.parvar]
        keys = [k for k in self.parvar if k]
        
        if self.independent:
            combos = np.array(np.meshgrid(*pars)).reshape(len(keys),-1).T
        else:
            combos = np.vstack(pars).T
            
        combo_dicts = [dict(zip(keys, combo)) for combo in combos]
        return combo_dicts

    @property
    def n(self):
        if self.independent:
            return np.prod([len(v) for v in self.parvar.values()])
        else:
            return len(list(self.parvar.values())[0])

    # Alternative constructor
    @classmethod
    def from_json(cls, json_file, **kwargs):
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
        return self.combos[self.ix]
    
    def get_next_combo(self):
        self.iterate_ix()
        combo = self.combos[self.ix]
        return combo
            
    def iterate_seastate(self):
        combo = self.get_next_combo()
        if combo is not None:
            for key in combo:
                setattr(self.seastate, key, combo[key])
            
        return self.seastate
    
    def get_next_seastate(self):
        self.iterate_seastate()            
        return self.seastate
      
    def iterate_ix(self):
        if self.ix == (self.n-1):
            self.ix = 0     #reset
        else:
           self.ix += 1
        
    def reset_ix(self):
        self.ix = 0