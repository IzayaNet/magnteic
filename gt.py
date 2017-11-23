#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 13:17:07 2017

@author: stan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:31:57 2017

@author: stan
"""



from sympy.solvers import solve as slv
from sympy import Symbol, Matrix, Function
import scipy as sc
import math as m
from scipy import optimize
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

E=np.identity(3) # Unit matrix definition
a1=3e-16
#K1=-5.421e-3*79.5774715 #J/cm^3
t1=10e-3
mu0=1.257e-6
Kc1=-610.0
lam=10e-1
z=np.array([0,0,1.0])
B=np.array([0,0,1.0])
M0=np.array([0,0,140000.0])
H0=B/mu0
h=np.array([0,0,0])
mv=np.array([0,0,100])
H0s=m.sqrt(H0[0]**2+H0[1]**2+H0[2]**2)
Ms=m.sqrt(M0[0]**2+M0[1]**2+M0[2]**2)
Z0=H0s/Ms
H0k=-(4*Kc1/(mu0*Ms))*np.array([0,0,1.0])
Zk=np.dot(H0k,z)/Ms
Zx=M0x=np.array([[0, -z[2], z[1]],
          [z[2], 0, -z[0]],
          [-z[1], z[0], 0]])
Na=-(4*Kc1/(mu0*Ms**2))*np.tensordot(z,z,axes=0)
#k_n=np.linspace(1e6,100e6,100)
w_k=np.zeros((100),dtype=complex)
Y=28e9*mu0
wM=Y*Ms
M0M0=Ms**2
M0t=np.tensordot(M0,M0,axes=0)
w_k_x=np.zeros((100,100),dtype=complex)
w_k_y=np.zeros((100,100),dtype=complex)
w_k_z=np.zeros((100,100),dtype=complex)
w_n=np.mgrid[1e10:100e9:100j]
k_n=np.mgrid[1e6:100e6:100j]
# External field
Ey=4000
Ez=60.0
w0=10e9
k0=10e9
t=1.0
x=1.0
z=0.0
def det0(W,k,w,dem):
    Heff=(a1*(1j*k)**2)*E+Na-(Z0+Zk)*E
    R=np.dot(Zx,Heff)
    L=1j*W*E
    A=R-L
    RelB=(Heff)*M0M0 
    RelA=np.dot(M0t,Heff)

    L=A-(lam/Ms**2)*(RelA-RelB)
    T=np.linalg.inv(L)
    #    H field computation
#    Ef=np.array([Ex*1j*np.exp(1j*(k0*x)),0,Ez*np.exp(1j*(k0*x))],dtype=complex)
#    Ef_c=np.conjugate(Ef)
#    h=np.cross(Ef,Ef_c)*np.cos(w0/wM*t)
#    h=np.cross(Ef,Ef_c)*np.exp(1j*(w0/wM*t))
    h=[0,gt(w,30,5),0]
    mv=np.dot(T,h)
#    mv=solve(L,F)
#    print(RelA-RelB)
#    print(mv[1])
    return mv[dem]
#for i in range(100):
#    for r in range(100):
#        w_k_x[i,r]=det0(w_n[i]/wM,k_n[r],0)
#        w_k_y[i,r]=det0(w_n[i]/wM,k_n[r],1)
#        w_k_z[i,r]=det0(w_n[i]/wM,k_n[r],2)
#plt.pcolormesh(k_n, w_n, w_k_x.real)
#plt.show()
#plt.pcolormesh(k_n, w_n, w_k_y.real)
#plt.show()
t1=np.linspace(0,37,38)
t2=np.linspace(0,185,186)
m_w_x=np.zeros((np.size(t2)),dtype=complex)
m_w_y=np.zeros((np.size(t2)),dtype=complex)
m_w_z=np.zeros((np.size(t2)),dtype=complex)
m_t_x=np.zeros((np.size(t2)),dtype=complex)
m_t_y=np.zeros((np.size(t2)),dtype=complex)
m_t_z=np.zeros((np.size(t2)),dtype=complex)
#m_w_x_s=np.zeros((20),dtype=complex)
#m_w_y_s=np.zeros((20),dtype=complex)
#m_w_z_s=np.zeros((20),dtype=complex)
#U=np.zeros((10,10),dtype=complex)
#V=np.zeros((10,10),dtype=complex)
#X=np.mgrid[1:10:10j]
#Y=np.mgrid[1:10:10j]

def hf(ws):
    h_t=Ey*np.exp(-((t1-5)**2)/10)
#    h_t=Ey*np.cos(10e9/wM*t1)
    h_w=np.fft.rfft(h_t)
    return h_w[ws]
    
def gt(ws,rt,n):
    h_t1=np.zeros((np.size(t2)),dtype=complex)
    for z in range(n):
        h_t1+=Ey*np.exp(-((t2-rt*(z+1))**2)/10)
    h_w1=np.fft.rfft(h_t1)
#    plt.figure(3)
#    plt.title("precession")
#    plt.plot(t2, h_t1, label="x")
#    plt.show()
#    plt.clf()
#    plt.close()
    return h_w1[ws]
for w1 in range(np.size(t2)):
    m_w_x[w1]=det0(28e9/wM,20e6,w1,0)
    m_w_y[w1]=det0(28e9/wM,20e6,w1,1)
    m_w_z[w1]=det0(28e9/wM,20e6,w1,2)
    
#print(m_w_x)
m_t_x=np.fft.irfft(m_w_x)
m_t_y=np.fft.irfft(m_w_y)
m_t_z=np.fft.irfft(m_w_z)
#for i in range(38):
#    if i<38/2:
#        m_t_x[i]=m_t_x_s[i+19]
#        m_t_y[i]=m_t_y_s[i+19]
#        m_t_z[i]=m_t_z_s[i+19]
#    else:
#        m_t_x[i]=m_t_x_s[i-19]
#        m_t_y[i]=m_t_y_s[i-19]
#        m_t_z[i]=m_t_z_s[i-19]
plt.figure(1)
plt.title("x")
plt.subplot(211)
plt.plot(t2, m_t_x.real, label="x")
plt.subplot(212)
plt.plot(t2, m_t_y.real, label="y")

plt.legend(loc='upper left')
plt.show()


r=np.sqrt(m_t_x_s.real**2+m_t_y_s.real**2)
theta=np.arctan(m_t_y_s.real/m_t_x_s.real)
theta[:4]=theta[:4]+m.pi
#plt.figure(2)
#plt.title("phase")
#plt.plot(t1,theta)
#plt.show()
#X=np.ones((1,1))
#Y=np.ones((1,1))
#plt.figure(3)
#plt.title("precession")
#plt.polar(theta[:22],r[:22])
#plt.show()
#plt.clf()
#plt.close()
#for t1 in range (200):
#    
##    Uf=np.fft.fft(U)
##    Vf=np.fft.fft(V)
#    plt.clf()    
#    plt.axes([0.025, 0.025, 0.95, 0.95])
#    plt.quiver(X, Y, m_t_x[t1].real, m_t_y[t1].real.real, alpha=.3, angles='xy', scale_units='xy', scale=1)
##plt.quiver(X, Y, U, V, edgecolor='k', facecolor='None', linewidth=.3)
#
#    plt.xlim(0, 11)
#    plt.xticks(())
#    plt.ylim(-1, 11)
#    plt.yticks(())
##    plt.show()
#    plt.savefig("m-at-time-%i.png"%t1,pad_inches=0.02, bbox_inches='tight')
#    plt.draw()
##    plt.show()
#    plt.clf()
#    plt.close()            
