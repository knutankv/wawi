# Include packages as standard
# from .general import *
# from wawi import abq

# Other packages
import numpy as np
import shutil
import csv
from copy import deepcopy

#
#
# HOW TO USE MODULE IMPORT IN __init__.py FILE.
#
#
# Unpack all functions in module to base level (package.*):
# from module import *
#
# Unpack specific function in module to base level (package.function):
# from module import function
#
# Import full module, and keep module structure (package.module.function):
# import module
#
# The __all__ array defines what modules are imported when from package import *
# is used.
#
# If possible conflicts with global names/packages, use dot:
# from .module import function/*
#
