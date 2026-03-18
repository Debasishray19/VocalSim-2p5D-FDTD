#!/usr/bin/env python3
import matplotlib.pyplot as plt

# Initialize lists
y_vals = []
z_vals = []

# Read data
with open('contour_coordinates.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 2:
            y_vals.append(float(parts[0]))
            z_vals.append(float(parts[1]))

# Plot
plt.figure(figsize=(8, 5))
plt.plot(y_vals, z_vals, marker='o', linestyle='-')
plt.xlabel('yVal')
plt.ylabel('zVal')
plt.title('Shape of Cross-section')
plt.grid(True)
plt.tight_layout()
plt.show()