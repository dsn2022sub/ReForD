from pyecharts.charts import Scatter3D, Page
# from pyecharts import
import random
import matplotlib
import matplotlib.pyplot as plt
from numpy import *
import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as ticker


def str2sec(x):
    h, m, s = x.strip().split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


flag = 0
ba = 0
lo = []
tl = []
mf = 0
bm = 0
ml = []
with open('3.txt', 'r') as tp_fh:
    for i, line in enumerate(tp_fh):
        if 'load average' in line and flag == 0:
            flag = 1
            ba = float(line.split(" ")[-1][:-1])
            continue
        if 'load average' in line and flag == 1:
            t = float(line.split(" ")[-1][:-1])
            lo.append(t - ba)
            tl.append(line.split(" ")[2])
        if 'KiB Mem :' in line and mf == 0:
            bm = int(line.split(" ")[9])
            mf = 1
        if 'KiB Mem :' in line and mf == 1:
            for a, kk in enumerate(line.split(" ")):
                if 'used' in kk:
                    t = int(line.split(" ")[a - 1])
                    ml.append(t - bm)

bt = str2sec(tl[0])
ttl = []
for i in tl:
    t = str2sec(i)
    ttl.append(t - bt)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(32, 24), dpi=800)
ax = plt.gca()

lo = [i * 100 for i in lo]
ml = [int(i / (1024 * 1024)) for i in ml]

_xtick_labels = ttl

plt.plot(ttl, lo
         , label="sys",
         color="g",
         linestyle='-',
         linewidth=20,
         marker='s',
         ms=4,
         alpha=0.8)

print(lo)
print('ha')
print(ttl)
print(ml)
print('ava cpu' + str(np.mean(lo)))
print('ava mem' + str(np.max(ml)))

plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(50))
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.spines['bottom'].set_linewidth(5);
ax.spines['left'].set_linewidth(5);
ax.spines['right'].set_linewidth(5);
ax.spines['top'].set_linewidth(5);

# plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(2500))
plt.xlabel("Time(seconds)", fontsize=64, fontdict={'family': 'Times New Roman'})
plt.ylabel("%CPU Utilization", fontsize=64, fontdict={'family': 'Times New Roman'})
# plt.ylabel("Memory Usage(GB)", fontsize=64,fontdict={'family': 'Times New Roman'})
# plt.ylabel("Memory Usage(GB)", fontsize=64,fontdict={'family': 'Times New Roman'})
plt.tick_params(axis='both', labelsize=48)
plt.ylim(0, 100)

# plt.axhline(y=1320/1024,linestyle=(3,(5,3)),lw=13)
plt.axhline(y=16, linestyle=(3, (5, 3)), lw=13)

# plt.show()
plt.savefig('2.jpg', dpi=800)
