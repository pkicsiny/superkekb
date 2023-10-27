from IPython.core.magic import Magics, magics_class, line_magic
"""
https://twitter.com/michaelwaskom/status/910590996509024256
"""

@magics_class
class Imports(Magics):

    @line_magic
    def imports(self, opts):

        lines = []

        lines.extend([
            "import os",
            "import numpy as np",
            "import pandas as pd",
            "from scipy import constants as cst",
            "from scipy import stats as st",
            "import scipy",
        ])

        if "xsuite" in opts:
            lines.extend([
            "import xobjects as xo",
            "import xtrack as xt",
            "import xfields as xf",
            "import xpart as xp",
            ])

        if "src" in opts:
            lines.extend([
            "import harmonic_analysis as ha",
            "import input_files.config as config",
            "import src.utils as utils",
            "import src.profiler as profiler",
            "import src.plot_utils as plot_utils",
            "import src.fma as fma",
            "import src.init as init",
            "import src.log as log",
            ])

        if "plot" in opts:
            lines.extend([
            "import matplotlib",
            "from matplotlib import pyplot as plt",
            "from matplotlib.ticker import FormatStrFormatter",
            "from matplotlib.ticker import ScalarFormatter",
            "from matplotlib.colors import LogNorm",
            "from matplotlib.pyplot import cm",
            "matplotlib.rcParams['font.size'] = 32",
            "matplotlib.rcParams['figure.subplot.left'] = 0.18",
            "matplotlib.rcParams['figure.subplot.bottom'] = 0.16",
            "matplotlib.rcParams['figure.subplot.right'] = 0.92",
            "matplotlib.rcParams['figure.subplot.top'] = 0.9",
            "matplotlib.rcParams['figure.figsize'] = (12,8)",
            "matplotlib.rcParams['font.family'] = 'Times New Roman'",
            "from scipy.stats import linregress",
            ])
            
        if "madx" in opts:
            lines.extend([
            "from cpymad.madx import Madx",
            ])
            
        if "pysbc" in opts:
            lines.extend([
            "import PySBC",
            ])
            
        self.shell.set_next_input("\n".join(lines), replace=False)


ip = get_ipython()
ip.register_magics(Imports)