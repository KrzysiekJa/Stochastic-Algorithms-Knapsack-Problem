import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np


size = [10, 100, 1000, 10000]
time_numpy2 = [9.07379999999236e-05, 0.0028796470000003183, 6.147212039999999, 12172.305457653]
value_dp = [51, 5384, 466567, 49571635]
time_m  = [0.01980683599999999, 0.2048368511799999, 0.8569423148599995, 5.125439968259998]
value_m = [50.94, 5289.76, 367048.88, 29500138.44]
time_G  = [0.006937316700000045, 0.008664259319999932, 0.008115169899999941, 0.018264451079958235]
value_G = [51.0, 5000.0, 292346.4, 26426515.08]



fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set(xlabel='problem size', ylabel='time [s]')
ax1.set_title('Time performance')
ax1.plot(size, time_numpy2, label="dynamic")
ax1.plot(size, time_m, label="PSO")
ax1.plot(size, time_G, label="genetic")
ax2.set_yscale('log')
ax2.set(xlabel='problem size')
ax2.set_title('Output values')
ax2.plot(size, value_dp, label="dynamic")
ax2.plot(size, value_m, label="PSO")
ax2.plot(size, value_G, label="genetic")

plt.legend()
plt.show()