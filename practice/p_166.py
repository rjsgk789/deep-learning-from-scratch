import numpy as np
import warnings

#warnings.filterwarnings("ignore", category = np.VisibleDeprecationWarning)
#x = np.array([[1.0, -0,5], [-2.0, 3.0]])   #ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ
x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print(x)

mask = (x <= 0)

print(mask)

out = x.copy()
print(out)
out[mask] = 0
print(out)

# this is RELU !!!!!!!!!!!!!!!!!!!