import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import configparser

# NeuroVoz DDK, PCGITA TDU, IPVS SP
Hubert0 = [.836, .793, .916]

Hubert1 = [.756, .8, .779]

labs = ['NeuroVoz\nDDK','PC-GITA\nTDU', 'IPVS\nSP']
indices = np.arange(len(Hubert0))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(6, 4))

colors = plt.cm.tab20b(np.linspace(0, 1, 20))

bar1 = ax.bar(indices, Hubert0, width, label='HuBERT-0', capsize=5, color=colors[14], alpha=0.7)
bar2 = ax.bar(indices + width, Hubert1, width, label='HuBERT-1', capsize=5, color=colors[12], alpha=0.7)

ax.set_ylim((0.5, 1))
ax.set_xlabel('Data set and speech task')
ax.set_ylabel('Model performance (AUC)')
ax.set_title('Comparison of embeddings on various datasets and speech tasks')
ax.set_xticks(indices + width/2, labels=labs)
# sec = ax.secondary_xaxis(location=0)
# sec.set_xticks([0.5, 1.5, 2.5], labels=['\nNeuroVoz', '\nPC-GITA', '\nIPVS'])
# sec.xaxis.set_ticks_position('none') 
# ax.xaxis.set_ticks_position('none') 
ax.legend()
plt.tight_layout()
path = r"C:\Users\INDYD\Dropbox\Uni MSc AI\Master_thesis_RAIVD\imgs"
if os.path.exists(path): # Only when running on laptop
    plt.savefig(os.path.join(path,"monolingual_comp_plot.pdf"))
plt.show()