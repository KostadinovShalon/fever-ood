from matplotlib import pyplot as plt
import numpy as np

# Data
distance = np.linspace(0, 10, 100)
energy = 1 / (1 + distance ** 2)
energy2 = 1 / (1 + distance ** 2 / 10)
energy3 = np.ones_like(distance)
# Plot
plt.plot(distance, energy, label='Normal Direction')
plt.plot(distance, energy2, label='Least singular vector Direction', color='orange')
plt.plot(distance, energy3, label='Null Space Direction', color='fuchsia')
plt.title('Change in Energy')
plt.xlabel('Distance')
plt.legend()
plt.ylim(0, 1.2)
plt.xlim(0, 10)
plt.ylabel('Energy')
plt.xlabel('Distance to an in-distribution sample')
# Remove x and y ticks
plt.xticks([])
plt.yticks([])
# Show plot
plt.show()
