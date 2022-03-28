import sys
import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import cm

def integrate(ic, ti, p):
	ic_list = ic
	m, k, xeq, rho, cd, ar = p
	
	x = []
	v = []
	for i in range(m.size):
		x.append(ic_list[2 * i])
		v.append(ic_list[2 * i + 1])

	sub = {}
	for i in range(m.size):
		sub[M[i]] = m[i]
		sub[K[i]] = k[i]
		sub[XEQ[i]] = xeq[i]
		sub[X[i]] = x[i]
		sub[Xdot[i]] = v[i]
		sub[Ar[i]] = ar[i]		
	sub['RHO'] = rho
	sub['CD'] = cd

	diff_eq = []
	for i in range(m.size):
		diff_eq.append(v[i])
		diff_eq.append(A[i].subs(sub))

	print(ti)

	return diff_eq

#---SymPy Derivation (also sets N parameter - number of masses/springs)

N = 3

RHO, CD, t = sp.symbols('RHO CD t')
Ar = sp.symbols('Ar0:%i' %N)
M = sp.symbols('M0:%i' %N)
K = sp.symbols('K0:%i' %N)
XEQ = sp.symbols('XEQ0:%i' %N)
X = dynamicsymbols('X0:%i' %N)

Xdot = []
T = 0
for i in range(N):
	Xdot.append(X[i].diff(t, 1))
	T += M[i] * Xdot[i]**2
T *= sp.Rational(1, 2)

V = K[0] * (X[0] - XEQ[0])**2
for i in range(1, N):
	V += K[i] * (X[i] - X[i-1] - XEQ[i])**2
V *= sp.Rational(1, 2)

L = T - V

A = []
Fc = sp.Rational(1, 2) * RHO * CD
for i in range(N):
	dLdX = L.diff(X[i], 1)
	dLdXdot = L.diff(Xdot[i], 1)
	ddtdLdXdot = dLdXdot.diff(t, 1)
	F = Fc * Ar[i] * sp.sign(Xdot[i]) * Xdot[i]**2
	dL = ddtdLdXdot - dLdX + F
	sol = sp.solve(dL, X[i].diff(t, 2))
	A.append(sp.simplify(sol[0]))

#--------------------------------------------------------

#----parameters, SciPy integration (intgration function at top)
#----and energy calculations----------------------------------

mass_a, mass_b = [1, 2]
k_a, k_b = [50, 100]
xeq_a, xeq_b = [1.5, 1.5]
xo_a, xo_b = [2, 6]
vo_a, vo_b = [0, 0]
rho = 1.225
cd = 0.47
rad = 0.25
tf = 60 
nfps = 30


initialize = "increment"


if initialize == "increment":
	m = np.linspace(mass_a, mass_b, N)
	k = np.linspace(k_a, k_b, N)
	xeq = np.linspace(xeq_a, xeq_b, N)
	xo = np.linspace(xo_a, xo_b, N)
	vo = np.linspace(vo_a, vo_b, N)
elif initialize == "random":
	rng=np.random.default_rng(92314311)
	m = (mass_b - mass_a) * np.random.rand(N) + mass_a
	k = (k_b - k_a) * np.random.rand(N) + k_a
	xeq = (xeq_b - xeq_a) * np.random.rand(N) + xeq_a
	xo = (xo_b - xo_a) * np.random.rand(N) + xo_a
	vo = (vo_b - vo_a) * np.random.rand(N) + vo_a
else:
	sys.exit("Initialization Routine Not Found. Choices are increment or random. Pick One.")


mass_radius = "proportional"
mr = np.zeros(N)
if mass_radius == "uniform":
        mr[:] = rad
elif mass_radius == "proportional":
        mr[:] = rad*m[:]/max(m)
else:
        sys.exit("Mass Radius Initialization Routine Not Found. Choices are uniform or proportional. Pick One.")
ar = np.pi * mr**2


p = [m, k, xeq, rho, cd, ar]
ic = []
for i in range(N):
	ic.append(xo[i])
	ic.append(vo[i])

nframes = tf * nfps
ta = np.linspace(0, tf, nframes)

xv = odeint(integrate, ic, ta, args=(p,))

x = np.zeros((N, nframes))
ke = np.zeros(nframes)
pe = np.zeros(nframes)
for i in range(nframes):
	ke_sub={}
	pe_sub={}
	for j in range(N):
		x[j][i] = xv[i, 2 * j]
		ke_sub[M[j]] = m[j]
		ke_sub[Xdot[j]] = xv[i, 2 * j + 1]
		pe_sub[K[j]] = k[j]
		pe_sub[XEQ[j]] = xeq[j]
		pe_sub[X[j]] = x[j][i]
	ke[i] = T.subs(ke_sub)
	pe[i] = V.subs(pe_sub)

E = ke + pe

#----aesthetics, plot, animation---------------

fig, a = plt.subplots()

xmax = x.max() 
if x.min() < 0:
	xmin = x.min()
else:
	xmin = 0
yline = 0

xmax += 2*mr[N-1]
xmin -= 2*mr[0]
ymax = yline + 2 * max(mr)
ymin = yline - 2 * max(mr)

spring_constant_proportional = "y"
dl = np.zeros((N,nframes))
nl = np.zeros(N)
dl[0][:] = x[0][:] + mr[0]
nl[0] = int(np.ceil((max(dl[0]))/(2 * mr[0])))
for i in range(1,N):
	dl[i][:] = x[i][:] - x[i-1][:]
	nl[i] = int(np.ceil(max(dl[i])/(2 * mr[i])))
lsf = 1
if spring_constant_proportional == "y":
	lr = np.zeros(N)
	lr[:] = 1 / (k[:] / max(k))
	for i in range(N):
		nl[i] = int(lsf * lr[i] * nl[i])

nlmax = int(max(nl))
xl = np.zeros((N,nlmax,nframes))
yl = np.zeros((N,nlmax,nframes))
for i in range(nframes):
	l0 = (x[0][i]/nl[0]) 
	xl[0][0][i] = x[0][i] - mr[0] - 0.5*l0
	for k in range(1,int(nl[0])):
		xl[0][k][i] = xl[0][k-1][i] - l0
	for k in range(int(nl[0])):
		yl[0][k][i] = yline+((-1)**k)*(np.sqrt(mr[0]**2 - (0.5*l0)**2))
	for j in range(1,N):
		lj = (x[j][i]-x[j-1][i]-(mr[j]+mr[j-1]))/nl[j]
		xl[j][0][i] = x[j][i] - mr[j] - 0.5*lj
		for k in range(1,int(nl[j])):
			xl[j][k][i] = xl[j][k-1][i] - lj
		for k in range(int(nl[j])):
			yl[j][k][i] = yline+((-1)**k)*(np.sqrt(mr[j]**2 - (0.5*lj)**2))

clist = cm.get_cmap('gist_rainbow', N)
			
def run(frame):
	plt.clf()
	plt.subplot(211)
	for i in range(N):
		circle=plt.Circle((x[i][frame],yline),radius=mr[i],fc=clist(i))
		plt.gca().add_patch(circle)
	plt.plot([xl[0][int(nl[0])-1][frame],-mr[0]],[yl[0][int(nl[0])-1][frame],yline],'xkcd:cerulean')
	for i in range(N):
		plt.plot([x[i][frame]-mr[i],xl[i][0][frame]],[yline,yl[i][0][frame]],'xkcd:cerulean')
	for i in range(1,N):
		plt.plot([xl[i][int(nl[i])-1][frame],x[i-1][frame]+mr[i-1]],[yl[i][int(nl[i])-1][frame],yline],'xkcd:cerulean')
	for j in range(N):
		for i in range(int(nl[j])-1):
			plt.plot([xl[j][i][frame],xl[j][i+1][frame]],[yl[j][i][frame],yl[j][i+1][frame]],'xkcd:cerulean')
	plt.title("N-Tuple Horizontal Spring with Air Resistance (N=%i)" %N)
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([xmin,xmax])
	plt.ylim([ymin,ymax])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(212)
	plt.plot(ta[0:frame],ke[0:frame],'xkcd:red',lw=1.0)
	plt.plot(ta[0:frame],pe[0:frame],'xkcd:cerulean',lw=1.0)
	plt.plot(ta[0:frame],E[0:frame],'xkcd:bright green',lw=1.5)
	plt.xlim([0,tf])
	plt.title("Energy")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')

ani=animation.FuncAnimation(fig,run,frames=nframes)
writervideo = animation.FFMpegWriter(fps=nfps)
ani.save('ntuple_horizontal_spring_w_air_resistance.mp4', writer=writervideo)
plt.show()



 



