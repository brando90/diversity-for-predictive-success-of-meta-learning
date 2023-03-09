#%%

import matplotlib.pyplot as plt

# Extract data
x = [441, 1961, 3045, 9541, 32901, 121093, 1427525]
y1 = [-0.0533, -0.184, 0.0794, -0.131, 0.0401, -0.0588, -0.0166]
y2 = [0.0624, -0.1, -0.00121, -0.149, -0.0689, -0.145, -0.0559]

# Plot data
plt.plot(x, y1, label='USL-MAML5', marker='x')
plt.plot(x, y2, label='USL-MAML10', marker='x')

# Add labels and grid
plt.xlabel('Number of Parameters (micod)')
plt.ylabel('Effect Size Difference')
plt.title('Difference between USL vs MAML using effect size')
plt.grid(True)

# Show legend
plt.legend()

# Save plot to desktop
plt.savefig('/Users/brandomiranda/Desktop/effect_size_difference.png')

# Show plot
plt.show()

#%%

import matplotlib.pyplot as plt

# Extract data
x = [441, 1961, 3045, 9541, 32901, 121093, 1427525]
y1 = [-0.0533, -0.184, 0.0794, -0.131, 0.0401, -0.0588, -0.0166]
y2 = [0.0624, -0.1, -0.00121, -0.149, -0.0689, -0.145, -0.0559]

# Plot data
plt.plot(x, y1, label='USL-MAML5', marker='x')
plt.plot(x, y2, label='USL-MAML10', marker='x')

# Add labels and grid
plt.xlabel('Number of Parameters (micod)')
plt.ylabel('Effect Size Difference')
plt.title('Difference between USL vs MAML using effect size')
plt.grid(True)

# Set x-axis scale to logarithmic
plt.xscale('log')

# Show legend
plt.legend()

# Save plot to desktop
plt.savefig('/Users/brandomiranda/Desktop/effect_size_difference_log_linear.png')

# Show plot
plt.show()


