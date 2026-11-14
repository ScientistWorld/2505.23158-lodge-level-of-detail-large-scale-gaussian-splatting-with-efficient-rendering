import numpy as np
from redteam.common import load_images, replace_proposed, zero_stats
pred = load_images()
replace_proposed(np.zeros_like(pred), zero_stats(len(pred), -1.e12))
