import sys

import numpy as np
import torch

X = torch.load(sys.argv[1])
MTCO1 = X[:, :, 10843]
np.savetxt(sys.stdout, MTCO1.detach().cpu().numpy())
