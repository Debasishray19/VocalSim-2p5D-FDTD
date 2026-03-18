import numpy as np
import matplotlib.pyplot as plt

# Load data from the text file (assuming whitespace-separated columns)
data = np.loadtxt('norm_ug.txt')

# Extract columns
t_milli = data[:, 0]
norm_ug = data[:, 1]

# Select the first 33075 points
t_plot = t_milli[:33075]
ug_plot = norm_ug[:33075]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(t_plot, ug_plot, linewidth=1.5)
plt.ylim(0, 1.5)
plt.xlabel('Time [ms]')
plt.ylabel('Normalized Volume Velocity [m^3/s]')
plt.title('Normalized Volume Velocity vs. Time')
plt.grid(True)
plt.show()