import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import configparser

# NeuroVoz TDU, DDK. PCGITA TDU, DDK. IPVS TDU, DDK
IFM = [0.68, 0.62, 0.67, 0.64, 0.99, 0.70]

NIFM = [0.68, 0.72, 0.67, 0.65, 0.98, 0.93]

labs = ['TDU', 'DDK','TDU', 'DDK','TDU', 'DDK']
indices = np.arange(len(IFM))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(6, 4))

bar1 = ax.bar(indices - width/2, IFM, width, label='IFM', capsize=5, color='g', alpha=0.7)
bar2 = ax.bar(indices + width/2, NIFM, width, label='NIFM', capsize=5, color='r', alpha=0.7)

ax.set_ylim((0.5, 1))
ax.set_xlabel('\nData set')
ax.set_ylabel('Model performance (AUC)')
ax.set_title('Comparison of feature types on different data sets')
ax.set_xticks(indices, labels=labs)
sec = ax.secondary_xaxis(location=0)
sec.set_xticks([0.5, 2.5, 4.5], labels=['\nNeuroVoz', '\nPC-GITA', '\nIPVS'])
sec.xaxis.set_ticks_position('none') 
ax.xaxis.set_ticks_position('none') 
ax.legend()
plt.tight_layout()
plt.show()