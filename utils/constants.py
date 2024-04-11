import numpy as np
import torch
import sys
from termcolor import cprint

bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
print(f'Python version: {sys.version_info[0]}')

# NeurIPS style
cm = 1 / 2.54

cprint('running non-deterministic', color='red')
# np.random.seed(0)
## torch.use_deterministic_algorithms(True)
# torch.manual_seed(0)
print(torch.__version__)