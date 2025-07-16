"""
Patch for pandas_ta to work with newer numpy versions
"""
import numpy as np
import sys

# Add the missing NaN alias to numpy
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

# Make this patch available to all imports
sys.modules['numpy'] = np
