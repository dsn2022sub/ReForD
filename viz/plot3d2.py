#!/usr/bin/env pythonw
import matplotlib as mpl  # noqa
from mpl_toolkits.mplot3d import Axes3D  # noqa
import matplotlib.pyplot as plt
import numpy as np
mpl.style.use('seaborn')
from getf import *
from getr import *
import random


# ****************************************************************************
# *                               Create data                                *
# **************************************************************************** 
with open('filtered.txt') as fh:
    data = eval(fh.readline().strip())
with open('raw.txt') as fh:
    data1 = eval(fh.readline().strip())
    data1 = random.sample(data1,int(len(data1)/1500))
x1 =[]
y1 = []
z1 =[]
l1=[]
x2 =[]
y2 = []
z2 =[]
l2=[]
c1=[]
l=[]
c2=[]

x3 =[]
y3 = []
z3 =[]

x4 =[]
y4 = []
z4 =[]
print('begin')
for line in data:        

    if line[3]=='attack':
        x1.append(float(line[0]))
        y1.append(float(line[1]))
        z1.append(float(line[2]))
        
        
    if line[3] =='normal' and line[0]>=-10 and line[0]<=10:
        x2.append(float(line[0]))
        y2.append(float(line[1]))
        z2.append(float(line[2]))
        

print('1done')
for line in data1:        

    if line[3]=='attack':
        x3.append(float(line[0]))
        y3.append(float(line[1]))
        z3.append(float(line[2]))
        #c3.append('r')
        #l.append('attack')
    if line[3] =='normal' and line[0]>=-10 and line[0]<=10:
        x4.append(float(line[0]))
        y4.append(float(line[1]))
        z4.append(float(line[2]))
        #c2.append('b')
        #l.append('normal')

print('2done')
# ****************************************************************************
# *                                 Plot data                                *
# ****************************************************************************
fig = plt.figure(figsize=(8,14))
ax1 = fig.add_subplot(2,1,1, projection='3d',facecolor ='white')
ax0 = fig.add_subplot(2,1,2, projection='3d',facecolor ='white')
title_font = {"family" : "Times New Roman",'size': 14}
ax1.set_title('(a):Data Distribution before compression',loc='center',y=-0.05,va='bottom',ha='center',font_properties=title_font)
ax0.set_title('(b):Data Distribution after compression',loc='center',y=-0.05,va='bottom',ha='center',font_properties=title_font)
x1=np.array(x1)
y1=np.array(y1)
z1=np.array(z1)
plt.subplots_adjust(bottom=0.05,top=0.96,wspace = 0.05,hspace = 0.005 )
legend_font = {"family" : "Times New Roman",'size': 14,'weight':'bold'}

type1=ax0.scatter(x1, y1, z1, c='r',marker='o',label='attack')
type2=ax0.scatter(x2, y2, z2, c='b',marker='o',label='normal')
ax0.set_xlabel('x')
ax0.set_ylabel('y')
ax0.set_zlabel('z')
#ax0.legend((type1, type2), ("attack", "normal"), loc = 'best',shadow=True,title='Type',fancybox=True,fontsize=10,title_fontsize=10,frameon=True )
print("1done")

type3=ax1.scatter(x3, y3, z3, c='r',marker='o',label='attack')
type4=ax1.scatter(x4, y4, z4, c='b',marker='o',label='normal')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
#ax1.legend((type3, type4), ("attack", "normal"), loc = 'best',shadow=True,title='Type',fancybox=True,fontsize=10,title_fontsize=10,frameon=True)
handles, labels = plt.gca().get_legend_handles_labels()
legend= fig.legend(handles, labels,bbox_to_anchor=[0.5, 0.003],markerscale=2, loc='lower center',prop=legend_font,borderpad=0.5,labelspacing=0.5,title_fontsize=28,frameon=True,ncol=3,borderaxespad=0.,facecolor='white',edgecolor='black')
legend.get_frame().set_linewidth(1.0)
print("2done")
# # If we knew what angles we wanted to set, these lines will set it#
elev = 20
azim = 45
ax0.view_init(elev, azim)
ax1.view_init(elev, azim)
#plt.legend()
# Show the figure, adjust it with the mouse
plt.savefig('m3d.jpg')
#plt.show()

