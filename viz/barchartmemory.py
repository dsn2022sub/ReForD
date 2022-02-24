import numpy as np
import matplotlib.pyplot as plt

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from math import exp
from math import e

legend_font = {"family": "Times New Roman", 'size': 14}

labels = ['LogGC', 'CPR', 'GS', 'SD', 'Ours(with&without logs)']
a = [7.29, 16.43, 8.77, 23.02, 150.46]
b = [0, 0, 0, 0, 1515.09 - 150.46]
c = [0.88, 1.0, 1.0, 1.0]
d = [0.80, 0.96, 0.78, 0.87]

a1 = [7.55, 9.15, 9.07, 22.73, 60.57]
b1 = [0, 0, 0, 0, 62 - 60.57]
c1 = [1.0, 1.0, 1.0, 1.0]
d1 = [0.89, 1.0, 1.0, 1.0]

a2 = [6.94, 9.15, 9.22, 29.80, 23.40]
b2 = [0, 0, 0, 0, 98.74 - 23.40]
c2 = [0.96, 0.92, 1.0, 1.0]
d2 = [0.98, 0.92, 0.98, 0.98]

x = np.arange(len(labels))
width = 0.38
fig = plt.figure(figsize=(8, 18))
plt.ylim(bottom=0.1)

ax = plt.subplot(3, 1, 1)
ax1 = plt.subplot(3, 1, 2)
ax2 = plt.subplot(3, 1, 3)

rects1 = ax.bar(x, a, width, label='Ratio', color='#4F81BD')
rects2 = ax.bar(x, b, width, bottom=a, label='Ratio without logs', color='#C0504D')

rects5 = ax1.bar(x, a1, width, label='Ratio', color='#4F81BD')
rects6 = ax1.bar(x, b1, width, bottom=a1, label='Ratio without logs', color='#C0504D')

rects9 = ax2.bar(x, a2, width, label='Ratio', color='#4F81BD')
rects10 = ax2.bar(x, b2, width, bottom=a2, label='Ratio without logs', color='#C0504D')
yindex = [math.pow(e, i) for i in range(9)]

yindex1 = [i for i in range(9)]

ax.set_ylabel('Ratio', fontsize=12)
ax.set_xlabel('(a):SC-1', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_yticks(yindex)
ax.set_yscale('log', basey=e)
ax.set_yticks(yindex)
ax.set_ylim(bottom=1, top=math.pow(e, 8))
ax.set_yticklabels([r'$e^{}$'.format(i) for i in yindex1]);  # use LaTeX formatted labels

ax1.set_ylabel('Ratio', fontsize=12)
ax1.set_xlabel('(b):SC-2', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)

ax1.set_yscale('log', basey=e)
ax1.set_yticks(yindex[:6])
ax1.set_ylim(bottom=1, top=math.pow(e, 5))
ax1.set_yticklabels([r'$e^{}$'.format(i) for i in yindex1[:6]]);  # use LaTeX formatted labels

ax2.set_ylabel('Ratio', fontsize=12)
ax2.set_xlabel('(c):APT-1', fontsize=12)
# ax.set_title('这里是标题')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)

ax2.set_yscale('log', basey=e)
ax2.set_yticks(yindex[:6])
ax2.set_yticklabels([r'$e^{}$'.format(i) for i in yindex1[:6]]);  # use LaTeX formatted labels

ax2.set_ylim(bottom=1, top=math.pow(e, 5))


def autolabel(rects, ax):
    for index, rect in enumerate(rects):
        height = rect.get_height()
        mheight = height
        # print(height)
        mheight = round(mheight, 2)
        if ax != ax1 or index != len(rects) - 1:
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center')
        else:
            ax.annotate('{}'.format(height) + '(62)',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center')


def autolabel_alter(rects, ax, add):
    for index, rect in enumerate(rects):
        height = rect.get_height()
        if index == len(rects) - 1 and ax != ax1:
            height += add
            mheight = height
            mheight = round(mheight, 2)
            ax.annotate('{}'.format(mheight),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom')


autolabel_alter(rects2, ax, 150.46)
autolabel(rects1, ax)

autolabel(rects5, ax1)
autolabel_alter(rects6, ax1, 60.57)
autolabel(rects9, ax2)
autolabel_alter(rects10, ax2, 23.40)

handles, labels = plt.gca().get_legend_handles_labels()
legend = fig.legend(handles, labels, bbox_to_anchor=[0.5, 0.008], loc='lower center', prop=legend_font, borderpad=0.5,
                    labelspacing=0.5, title_fontsize=28, frameon=True, ncol=4, borderaxespad=0., facecolor='white',
                    edgecolor='black')
legend.get_frame().set_linewidth(1.0)
plt.subplots_adjust(bottom=0.1, top=0.95, wspace=0.1, hspace=0.15)

plt.savefig('barmemory.jpg')
plt.show()
