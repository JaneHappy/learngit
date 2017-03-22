# chapter 2

import numpy as np
import matplotlib.pyplot as plt

print "Please input the number:"
n = int(raw_input("n = "))

plt.figure(1)
# = np.linspace(1,10,1,endpoint=False)
for j in np.array(range(n))+1:
    for k in np.array(range(n))+1:
        if 1<=j<k+j<=n:
            plt.figure(1)
            plt.plot(j, k, '*')
            print "j ="+str(j)+", k ="+str(k)

plt.show()
