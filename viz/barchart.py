import numpy as np
import matplotlib.pyplot as plt

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

legend_font = {"family": "Times New Roman", 'size': 14}

labels = ['LogGC', 'CPR', 'GS', 'SD', 'Ours']

# ****************************************************************************
# *                               Create data                                *
# ****************************************************************************
a = [0.78, 0.96, 0.72, 0.86, 1.0]
b = [0.73, 0.92, 0.64, 0.78, 1.0]
c = [0.88, 1.0, 1.0, 1.0, 1.0]
d = [0.80, 0.96, 0.78, 0.87, 1.0]

a1 = [0.88, 1.0, 1.0, 1.0, 1.0]
b1 = [0.80, 1.0, 1.0, 1.0, 1.0]
c1 = [1.0, 1.0, 1.0, 1.0, 1.0]
d1 = [0.89, 1.0, 1.0, 1.0, 1.0]

a2 = [0.98, 0.92, 0.98, 0.98, 1.0]
b2 = [1.0, 0.92, 0.96, 0.96, 1.0]
c2 = [0.96, 0.92, 1.0, 1.0, 1.0]
d2 = [0.98, 0.92, 0.98, 0.98, 1.0]

x = np.arange(len(labels))
width = 0.1
fig = plt.figure(figsize=(8, 14), dpi='1500')
# fig, ax = plt.subplots()

ax = plt.subplot(3, 1, 1)
ax1 = plt.subplot(3, 1, 2)
ax2 = plt.subplot(3, 1, 3)

rects1 = ax.bar(x - width * 1.5, a, width, label='Accuracy', color='#4F81BD')
rects2 = ax.bar(x - width * 0.5, b, width, label='Precision', color='#C0504D')
rects3 = ax.bar(x + width * 0.5, c, width, label='Recall', color='#9BBB59')
rects4 = ax.bar(x + width * 1.5, d, width, label='F1-Score', color='#9F4C7C')

rects5 = ax1.bar(x - width * 1.5, a1, width, label='Accuracy', color='#4F81BD')
rects6 = ax1.bar(x - width * 0.5, b1, width, label='Precision', color='#C0504D')
rects7 = ax1.bar(x + width * 0.5, c1, width, label='Recall', color='#9BBB59')
rects8 = ax1.bar(x + width * 1.5, d1, width, label='F1-Score', color='#9F4C7C')

rects9 = ax2.bar(x - width * 1.5, a2, width, label='Accuracy', color='#4F81BD')
rects10 = ax2.bar(x - width * 0.5, b2, width, label='Precision', color='#C0504D')
rects11 = ax2.bar(x + width * 0.5, c2, width, label='Recall', color='#9BBB59')
rects12 = ax2.bar(x + width * 1.5, d2, width, label='F1-Score', color='#9F4C7C')

# rects5 = ax.bar(x + width*2 + 0.04, e, width, label='e')


ax.set_ylabel('Rate', fontsize=12)
ax.set_xlabel('(a):SC-1', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels)

ax1.set_ylabel('Rate', fontsize=12)
ax1.set_xlabel('(b):SC-2', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)

ax2.set_ylabel('Rate', fontsize=12)
ax2.set_xlabel('(c):APT-1', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(labels)


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        # ax.annotate('{}'.format(height),
        #            xy=(rect.get_x() + rect.get_width() / 2, height),
        #            xytext=(0, 3),
        #            textcoords="offset points",
        #            ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6)
autolabel(rects7)
autolabel(rects7)
autolabel(rects8)
autolabel(rects9)
autolabel(rects11)
autolabel(rects12)

handles, labels = plt.gca().get_legend_handles_labels()
legend = fig.legend(handles, labels, bbox_to_anchor=[0.5, 0.004], loc='lower center', prop=legend_font, borderpad=0.5,
                    labelspacing=0.5, title_fontsize=28, frameon=True, ncol=4, borderaxespad=0., facecolor='white',
                    edgecolor='black')
legend.get_frame().set_linewidth(1.0)
plt.subplots_adjust(bottom=0.1, top=0.95, wspace=0.1, hspace=0.15)

plt.savefig('bar.jpg')
plt.show()
