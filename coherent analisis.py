# -*- coding: utf-8 -*-
# @author: Santiago Villodre Martinez 

'Declaracion de librerias'
import os 
import time
import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.special import spherical_jn
from scipy.special import gamma
from scipy.integrate import simpson
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy import signal
from matplotlib.lines import Line2D


################################################################################################################################  
'Selección de variables comunes'
################################################################################################################################

'Condicionantes a la aproximación'
real = False              # Si real = True, tomo valores int en la comparacion
origenT = True            # Comenzamos con los valores de T directamente 
cortarPE = False          # Realizamos un corte en bin_corte 
T_efici = False           # Añade la eficiencia en tiempos 
ff = False                # Añade el factor de forma 

'Condicionantes al programa'
pantalla = True             # Mostrar los resultados por pantalla
Graf = True                 # Realizar los graficos
times = True                # Imprimimos tiempo de ejecucion
valsigma = False            # Calcula los valores para 1 sigma 
save = False                # Guardamos las graficas directamente
saveZ = False                # Guarda los valores de la funcion CHI 
np.random.seed(48)              
start_time = time.time()


################################################################################################################################  
'Selección de calculos'
################################################################################################################################
'Primeras graficas y calculos'
grafflux = False           # Realiza un plot del flujo 
graf_time = False          # Gráfica de los coeficientes temporales
N_eventos_naive = False    # Calcula el numero de eventos Naive 

'Grafica de DN/DT y DN/DPE'
graf2q = False             # Gradica de DN/DT 
grafTee = False            # Grafica la relacion entre T_nr y PE
graf2q_Tee = False         # Grafica de DN/DPE 

'Grafica del Chi con primera aproximación'
grafchi1 = False            # Grafica de chi cuadrado 
grafchi2q = False           # Grafica de elipse de chi cuadrado 
grafmini1d = False          # Grafica de chi2 dos q, minimizando una dimension
grafchi2q_3d = False        # Grafica de chi2 en tres dimensiones
graftopo = False            # Genero un mapa topologico con las funciones chi

'Gráficas de correlación con primera aproximación'
graffcorr = False           # Grafica de chi2 con una conbinacion lineal 

'Gráficas de los factores de forma'
graf_FF = False             # Factor de forma
grafJacob = False           # Grafica de los jacobianos
grafresol = False           # Calculo de la resolución energetica 
grafefici = True           # Grafica de las eficiencias

'Calculo del numero de eventos con aproximación final'
grafmasterdatqa = False     # Calcula el numero de eventos y lo guarda (Poner corte)
grafbins = False            # Calcula el histograma de eventos en t y PE
grafdatacomp = False        # Comparativa del numero de eventos (Quitar cortarPE)

'Grafica del Chi con aproximación final'
grafchieffi = False         # Grafica de chi con y sin eficiencias
grafmarefi = False          # Marginaliza el nuevo chi 
grafchibi = False           # Grafica del chi con distribucion temporal y energetica
grafmarbi = False            # Grafica de la marginalización con temporal y enegetica
grafchicobi = False         # Grafica comparativa entre tamaños al meter info temporal

'Gráficas de correlación con la aproximación final'
grafcorrefici = False       # Grafica de correlacion con las eficiencias
grafcorbi = False            # Grafica de la correlación con temporal y enegetica

################################################################################################################################
################################################################################################################################  
'Valores conocidos y necesarios para aproximaciones'
################################################################################################################################
################################################################################################################################

#Masas
m_pi = 139.57018            # Mev
m_mu = 105.6583715          # Mev
m_det = 14.6                # Masa activada del detector en kg
m_cs_atomica = 132.90549    # Masa atomica del cesio en umas 
m_i_atomica = 126.90447     # Masa atomica del ido en umas

# Numero de particulas
n_pot = 31.98 * 1e22        # Numero de protones 
f_nu_p = 0.0848             # numero de particulas por foton 
Z = 54                      # Numero de protones en el nucleo 
N = 130                     # Numero de nucleones en el nucleo

# Parametros experimento 
T0 = .00                    # Energia cinetica en Mev
T1 = .01                    # Energia cinetica en Mev
L_m = 19.3                  # Distancia a la que se mide en m 
LY = 13.35*1e3              # Constante de relacion entre PE Y QT en MeV-1

time0 = 0.                  # Tiempo inicial del experimento
timef = 6.                  # Tiempo final del experimento
tau_mu = 2.196981           # Semivida del antimuon 
tau_pi = 2.6033e-2          # Semivida del pion 

# Constantes 
v = 2.4622 * 1e5                # Constante de acoplamiento en MeV 
E_nu = (m_pi**2-m_mu**2)/2/m_pi # Energia del neutrino en la desintegracion 
h = 4,135 * 1e9                 # Constante de Plank en Mev s 
cl = 2.99792458*1e8             # Velocidad de la luz en m/s
gf = 1.16637*1e-5               # Constante de Fermi 

# Radios atomicos
ns = 0.9*1e-15                  # Nuclear Skin en fm 
R_p = (4.821 + 4.766)/2*1e-15   # Radio de los protones en fm
R_n = (5.09+ 5.03)/2 *1e-15     # Radio de los neutrones en fm 

# Cargas debiles cuando hay dos 
Q_2_SM_mu = 7180.286802266689
Q_2_SM_ee = 2393.4289340888963

################################################################################################################################  
'Selección de parametros'
################################################################################################################################
# Bineado de la simulación 
num_bins = 52                           # Si CortePe, esto se pondra en 52
num_bins_temp = 12                      # Se ajusta luego para paso de 0.5
bin_corte = 4                           # Bin en el que se corta para graficar (4 para graf_bins)
time_max = 6                            # Valor temporal maximo 

# Factores para las representaciones de 1 sigma                                                  
f_e = 0.05                                 # Factor de escala para funcion Chi 
fu,fe = 0.4,0.8                            # Factores de escala para qe y qu
ffu_cl,ffe_cl = 1.2,1.5                    # Factores de escala para com. lin
ffu_ef,ffu_ef = 0.3,0.6                    # Factores de escala para chi efic
ffu,ffe = 1.5,2.8                          # Factores que poner si aproxfinal
ffubi,ffebi = 0.4,0.62                     # Factores para distribución bi

# Separaciones de bins
splitred = 1500                            # Divisiones de Q para chi 2 dim 
split = 100000                             # Divisiones de Q para chi 1 dim 
t_split = 10000                            # Divisiones de PE 
t_split_rec = 1000                         # Divisiones de PE  2500
timepass = 24000                           # Divisiones de t 10000
num_split_bins = 1000                      # Divisiones de T 
tol = 1e-2                                 # Tolerancia en 1 sigma

# Otros valores del chi 
corte = 1.                                 # Valor de numero minimo de eventos
sigmalegend = 1                            # Valor de sigma en la leyenda
sigma = 2.3                                # VAlor de sigma en bidimensional
#sigma = 2.3                               # VAlor de sigma en unidimensional

# Valores del parametro de nuissance 
alphamax = 0.20
splitredaph = 100
Error_alpha = 0.01


################################################################################################################################
################################################################################################################################
"Definicion de funciones"
################################################################################################################################
################################################################################################################################

################################################################################################################################
'Funcion auxiliar para graficar y dar resultados '
################################################################################################################################

class MathTextSciFormatter(mticker.Formatter):
    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt
    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)

################################################################################################################################
"Conversiones "
################################################################################################################################

def J_Mev (j):
    return j*6.2415*1e12

def uma_kg(m):
    return m*1.66053886*1e-27 

def uma_MeV(m):
    return m*931.49410242

def m_csi_uma(z,n):
    return (z*1.0072764592522818 + (n-z)*1.008664907871162)

def m_to_Mev(d):
    return d*1/197.3269804*1e15

def q(t):
    return (t*2*mn)**0.5  

################################################################################################################################
"Calculo de las cargas"
################################################################################################################################

# Calculo del valor de la carga debil en el SM 
def Q(z,n):
    return (z*0.0229  - 1.00*(n-z))**2

# Calculo de la carga debil con las correcciones radiales 
def Qmu(z,n):
    return (z*0.0582 - 1.02352*(n-z))**2

# Calculo de la carga debil con las correcciones radiales 
def Qe(z,n):
    return (z*0.0747 - 1.02352*(n-z))**2

################################################################################################################################
"Funciones de Flujos temporales y energeticos"
################################################################################################################################

# Flujo energetico de neutrinos mu 
def dphi_mu_dE_nu(npot,fnu,E_nu_pi,l):
    return (npot * fnu / (4 * np.pi * l**2)) 

# Flujo energetico de neutrinos e 
def dphi_ve_dE(E,npot,fnu,mmu,l):
    return (npot*fnu/(4*np.pi*l**2))*(192*E**2/mmu**3) *(1/2-E/mmu)

# Flujo energetico de neutrinos antimu
def dphi_vmu_dE(E, npot,fnu,mmu,l):
    return (npot * fnu/(4*np.pi*l**2))*(64*E**2/mmu**3)*(3/4-E/mmu)    

# Flujo temporal de neutrinos prompt         
def dP_dt_mu(t_list,taumu,taupi):
    apot,bpot = 0.44,0.15
    coef = 1/((2*np.pi)**0.5*bpot*taupi) 
    results = []
    for t in t_list: 
        P_mu = quad(lambda tp:np.exp(-(tp-apot)**2/(2*bpot**2))*np.exp(-(t-tp)/taupi),0,t)
        results.append(P_mu[0])
    results = np.array(results)
    return results*coef

# Flujo temporal de neutrinos delay 
def dP_dt_ee(t_list,taumu,taupi):
    coef = 1/taumu
    results = []
    for t in t_list: 
        tp = np.linspace(0, t,timepass)
        P_mu = P_mu_splin(tp)
        ddP = P_mu * np.exp(-(t - tp)/ taumu) 
        P_ee = simpson(ddP,tp)
        results.append(P_ee)
    results = np.array(results)
    return coef*results             

################################################################################################################################
"Funciones principales para el calculo del SM"
################################################################################################################################

def eq_promt(t,z,n,mdet,E,npot,fnu,Q,l,FF):    
    mn = uma_MeV(m_csi_uma(z, n))
    Nt = mdet/uma_kg(m_csi_uma(z, n))
    #Nt = 6.769819119818411e+27
    #mn = 122073.6637782
    tmax = 2*E**2/(mn + 2*E)
    dfi = npot * fnu/(4*np.pi*l**2)
    dsigma = (mn + t)/(8*v**4*np.pi)*(1 - (mn + 2*E)*t/(2*E**2))*Q*FF**2
    sol = Nt * dfi * dsigma 
    buenos = t <= tmax  
    return  sol*buenos 

def eq_promt_nq(t,z,n,mdet,E,npot,fnu,l,FF): 
    mn = uma_MeV(m_csi_uma(z, n))
    Nt = mdet/uma_kg(m_csi_uma(z, n))
    #Nt = 6.769819119818411e+27
    #mn = 122073.6637782
    tmax = 2*E**2/(mn + 2*E)
    dfi = npot * fnu/(4*np.pi*l**2)
    dsigma = (mn + t)/(8*v**4*np.pi)*(1 - (mn + 2*E)*t/(2*E**2))*FF**2
    sol = Nt * dfi * dsigma 
    buenos = t <= tmax  
    return  sol*buenos  

def eq_delay_u(t,z,n,mdet,Emax,npot,fnu,Q,m,l,FF):
    Nt = mdet/uma_kg(m_csi_uma(z, n))
    #Nt = 6.769819119818411e+27
    mn = uma_MeV(m_csi_uma(z, n))
    #mn = 122073.6637782
    Emin = t/2 + np.sqrt(t**2/4 + t*mn/2)
    t1 = (mn + t)/(32*np.pi**2*v**4*l**2)
    Asup = (Emax/m)**3 - (Emax/m)**4
    Ainf = (Emin/m)**3 - (Emin/m)**4
    Bsup = 4*Emax**3/(3*m**4) - 3*Emax**2/(2*m**3) - 3*Emax*mn/(2*m**3) + Emax**2*mn/(m**4)
    Binf = 4*Emin**3/(3*m**4) - 3*Emin**2/(2*m**3) - 3*Emin*mn/(2*m**3) + Emin**2*mn/(m**4)
    fsup = 16*(Asup + Bsup*t)
    finf = 16*(Ainf + Binf*t)
    return Nt* npot * fnu*t1*(fsup-finf)*Q*FF**2

def eq_delay_u_nq(t,z,n,mdet,Emax,npot,fnu,m,l,FF):
    Nt = mdet/uma_kg(m_csi_uma(z, n))
    mn = uma_MeV(m_csi_uma(z, n))
    #mn = 122073.6637782
    Emin = t/2 + np.sqrt(t**2/4 + t*mn/2)
    t1 = (mn + t)/(32*np.pi**2*v**4*l**2)
    Asup = (Emax/m)**3 - (Emax/m)**4
    Ainf = (Emin/m)**3 - (Emin/m)**4
    Bsup = 4*Emax**3/(3*m**4) - 3*Emax**2/(2*m**3) - 3*Emax*mn/(2*m**3) + Emax**2*mn/(m**4)
    Binf = 4*Emin**3/(3*m**4) - 3*Emin**2/(2*m**3) - 3*Emin*mn/(2*m**3) + Emin**2*mn/(m**4)
    fsup = 16*(Asup + Bsup*t)
    finf = 16*(Ainf + Binf*t)
    return Nt* npot * fnu*t1*(fsup-finf)*FF**2 

def eq_delay_e(t,z,n,mdet,Emax,npot,fnu,Q,m,l,FF):
    Nt = mdet/uma_kg(m_csi_uma(z, n))
    mn = uma_MeV(m_csi_uma(z, n))
    #Nt = 6.769819119818411e+27
    #mn = 122073.6637782
    Emin = t/2 + np.sqrt(t**2/4 + t*mn/2)
    t1 = (mn + t)/ (32*np.pi**2*v**4*l**2)
    Asup = 2*(Emax/m)**3 - 3*(Emax/m)**4
    Ainf = 2*(Emin/m)**3 - 3*(Emin/m)**4
    Bsup = 4*Emax**3/(m**4) - 3*Emax**2/(m**3) - 3*Emax*mn/(m**3) + 3*Emax**2*mn/(m**4)
    Binf = 4*Emin**3/(m**4) - 3*Emin**2/(m**3) - 3*Emin*mn/(m**3) + 3*Emin**2*mn/(m**4)
    fsup = 16*(Asup + (Bsup)*t)
    finf = 16*(Ainf + (Binf)*t)
    return Nt*npot*fnu*t1*(fsup-finf)*Q *FF**2 

def eq_delay_e_nq(t,z,n,mdet,Emax,npot,fnu,m,l,FF):
    Nt = mdet/uma_kg(m_csi_uma(z, n))
    #Nt = 6.769819119818411e+27
    mn = uma_MeV(m_csi_uma(z, n))
    Emin = t/2 + np.sqrt(t**2/4 + t*mn/2)
    t1 = (mn + t)/ (32*np.pi**2*v**4*l**2)
    Asup = 2*(Emax/m)**3 - 3*(Emax/m)**4
    Ainf = 2*(Emin/m)**3 - 3*(Emin/m)**4
    Bsup = 4*Emax**3/(m**4) - 3*Emax**2/(m**3) - 3*Emax*mn/(m**3) + 3*Emax**2*mn/(m**4)
    Binf = 4*Emin**3/(m**4) - 3*Emin**2/(m**3) - 3*Emin*mn/(m**3) + 3*Emin**2*mn/(m**4)
    fsup = 16*(Asup + (Bsup)*t)
    finf = 16*(Ainf + (Binf)*t)
    return Nt*npot*fnu*t1*(fsup-finf)*FF**2 

#Función que usa las anteriores para obtener Dn/DT
def dn_calculator(tlist,z,n,mdet,Enu,Emax,npot,f,Q2SM,nm,l,cargado,FF):
    if FF:
        mn = uma_MeV(m_csi_uma(z, n))
        q_2_array = (tlist*2*mn)**0.5
        F_p,qrop = Factos_nucleon(q_2_array,ns_mev,R_p_mev)
        F_n,qron = Factos_nucleon(q_2_array,ns_mev,R_n_mev)
        FF = 1/((N-Z) - 0.076*Z)*((N-Z)*F_n - 0.076*Z*F_p)
    else: 
        FF = 1.
        
    if cargado == True:
        promt= eq_promt(tlist,z,n,mdet,Enu,npot,f,Q2SM,l,FF)
        delay_e = eq_delay_e(tlist,z,n,mdet,Emax,npot,f,Q2SM,nm,l,FF)
        delay_u = eq_delay_u(tlist,z,n,mdet,Emax,npot,f,Q2SM,nm,l,FF)
        return promt+delay_e+delay_u,promt,delay_e,delay_u
     
    if cargado == False:
        promt = eq_promt_nq(tlist,z,n,mdet,Enu,npot,f,l,FF)
        delay_e = eq_delay_e_nq(tlist,z,n,mdet,Emax,npot,f,nm,l,FF)
        delay_u = eq_delay_u_nq(tlist,z,n,mdet,Emax,npot,f,nm,l,FF)
        return promt+delay_e+delay_u,promt,delay_e,delay_u
    
#Función que usa las anteriores para obtener Dn/DT con dos cargas
def dn_calculator2q(tlist,z,n,mdet,Enu,Emax,npot,f,nm,l,FF):
    if FF:
        mn = uma_MeV(m_csi_uma(z, n))
        q_2_array = (tlist*2*mn)**0.5
        F_p,qrop = Factos_nucleon(q_2_array,ns_mev,R_p_mev)
        F_n,qron = Factos_nucleon(q_2_array,ns_mev,R_n_mev)
        FF = 1/((N-Z) - 0.076*Z)*((N-Z)*F_n - 0.076*Z*F_p)
    else: 
        FF = 1.
        
    promt= eq_promt(tlist,z,n,mdet,Enu,npot,f,Q_2_SM_mu,l,FF)
    delay_e = eq_delay_e(tlist,z,n,mdet,Emax,npot,f,Q_2_SM_ee,nm,l,FF)
    delay_u = eq_delay_u(tlist,z,n,mdet,Emax,npot,f,Q_2_SM_mu,nm,l,FF)
    return promt+delay_e+delay_u,promt,delay_e,delay_u

# Funcion que da el numero de eventos
def valoresexp(t,z,n,mdet,Enu,Emax,npot,f,Q,mmu,l,FF):
    dn,dnp,dnde,dndu = dn_calculator(t,z,n,mdet,Enu,Emax,npot,f,Q,mmu,l,True,FF)
    dnnq,dnpnq,dndenq,dndunq = dn_calculator(t,Z,N,mdet,Enu,Emax,npot,f,Q,mmu,l,False,FF)
    np,nde,ndu = simpson(y=dnp, x = t),simpson(y=dnde,x = t),simpson(y=dndu, x = t)
    npnq,ndenq,ndunq = simpson(y=dnpnq, x = t),simpson(y=dndenq ,x= t),simpson(y=dndunq ,x= t)    
    return np,nde,ndu,npnq,ndenq,ndunq

################################################################################################################################
"Funciones para calcular la dependencia temporal y eficiencias"
################################################################################################################################

# Factor de forma nuclear
def Factos_nucleon(q,s,R):
    R_0 = (5/3*R**2 - 5*s**2 )**0.5
    q_R0 = q*R_0
    j_1 = spherical_jn(1,q_R0, derivative=False)
    j_1[0] = j_1[1]
    exp_s = np.exp(-(q**2*s**2)/2)
    return 3*j_1 /q_R0*exp_s, q_R0  

# Eficiencia temporal
def eft(t):
    dentro = t < 0.52 
    return 1*dentro + np.exp(-0.0494*(t-0.52))* np.logical_not(dentro) 

# Eficiencia energetica 
def efpe(pe):
    a,b,c,d = 1.32045,0.285979,10.8646,-0.33322
    r = (a/( 1+ np.exp(-b*((pe) - c))) + d) 
    dentro = r > 0.
    return r*dentro  

# Resolución energetica 
def energy_resolution_origin(PE, Tee):
    a = 0.0749/Tee 
    b = 9.56 * Tee
    prefactor = (a * (1 + b))**(1 + b) / gamma(1 + b)
    return prefactor * (PE**b) * np.exp(-a * (1 + b) * PE)        
    
################################################################################################################################
"Funciones para realizar los cambios de variable"
################################################################################################################################

def QF(T):    # De T_nr a Tee en MeV
    return (0.0554628*T**1 + 4.30681*T**2 -111.707*T**3 + 840.384*T**4)

def QFT(T,ly):  # De T_nr a PE en MeV
    return ly*(0.0554628*T**1 + 4.30681*T**2 -111.707*T**3 + 840.384*T**4)

def QF_k(T):    # De T_nr a Tee en KeV
    a,b,c,d = 0.0554628,4.30681*1e-3,-111.707*1e-6,840.384*1e-9
    return (a*T**1 + b*T**2 + c*T**3 + d*T**4)

def QFT_k(T):     # De T_nr a PE en KeV
    a,b,c,d = 0.0554628,4.30681*1e-3,-111.707*1e-6,840.384*1e-9
    return 13.35*(a*T**1 + b*T**2 + c*T**3 + d*T**4)

def D_QF_k(T):    # De T_nr a Tee en KeV
    a,b,c,d = 0.0554628,4.30681*1e-3,-111.707*1e-6,840.384*1e-9
    return (a + b*T*2 + 3*c*T**2 + 4*d*T**3)

def D_QFT_k(T):     # De T_nr a PE en KeV
    a,b,c,d = 0.0554628,4.30681*1e-3,-111.707*1e-6,840.384*1e-9
    return 13.35*(a + b*T*2 + 3*c*T**2 + 4*d*T**3)

################################################################################################################################
"Funciones principales para la integracion"
################################################################################################################################
# Función que realiza una integracion para cada uno de los bins 
def split_bins(bins,dn,tbins):
    N = np.zeros(bins)
    for i in range(bins):
        N[i] = simpson(y = dn[i], x = tbins[i])
    return N

def doble_integral(dn,Pebins,tray,R,EF,nib): # Tray en KeV
    dn = dn*1e-3                             # Lo ponemos en KeV
    if not(np.all(R == np.diag(np.diagonal(R)))):
        # La matriz R no es una delta, calculo la primera integrla y la pongo en funcion de TE
        R_t = np.multiply(R,J_T_TE.reshape(-1, 1))  # R(Pe,T)
        R_dn = np.multiply(R_t,dn.reshape(-1, 1))   # R(Pe,T)*DN/DT(T)
        IT = np.apply_along_axis(simpson, 0, R_dn, tray) # 
    else:
        # La matriz R es una delta, pongo dn funcion de PE
        IT = dn*J_PE_T
    return split_bins(nib,np.array_split(IT*EF,nib),Pebins)    
    
################################################################################################################################
"Funciones principales para el calculo del chi2"
################################################################################################################################
# Funcion chi2 en una dimension 
def chi_red(qr,Nr,Kr,Erred,pos):
    if pos == True:
        chi = np.add.reduce(2*(-Nr+ Kr*qr+Nr*np.log(Nr/(Kr*qr))), axis=1)
        
    if pos == False: 
        chi = np.add.reduce(((Nr - Kr*qr)/(Erred))**2, axis=1)
    return chi - np.min(chi)

# Funcion chi2 en dos dimensiones 
def chi_2q(q,N,K,Error,pos):
    if pos == True:
        chip = 0
        for i in range(len(N)):
            chip += 2*(-N[i]+ (K[0,i]*q[0]+K[1,i]*q[1]) + N[i]*np.log(N[i]/(K[0,i]*q[0]+K[1,i]*q[1])))
        return chip 
    
    if pos == False: 
        chip = 0
        for i in range(len(N)):
            chip += ((N[i] - (K[0,i]*q[0]+K[1,i]*q[1]))/(Error[i]))**2 
        return chip

# Chi2 de manera que esta listo para ser minimizado
def chi2mini(qx,qy,N,K,Error,pos):
    if pos == True:
        chip = 0
        for i in range(len(N)):
            chip += 2*(-N[i]+ (K[0,i]*qx+K[1,i]*qy) + N[i]*np.log(N[i]/(K[0,i]*qx+K[1,i]*qy)))
        return chip 
    
    if pos == False: 
        chip = 0
        for i in range(len(N)):
            chip += ((N[i] - (K[0,i]*qx +K[1,i]*qy))/(Error[i]))**2 
        return chip   

def chi_2q_bi(q,N,K,Error):
    chip = 0
    for j in range(len(N)):
        for i in range(len(N[0])): 
            chip += ((N[j,i] - (K[0,j,i]*q[0]+K[1,j,i]*q[1]))/(Error[j,i]))**2 
    return chip    
    
################################################################################################################################
'Funcion que calcula los puntos de la red que estan a un sigma'
################################################################################################################################

# Para el caso de 1 dimension
def puntossigmachi(chi,qrr):
    dis = np.abs(chi - (sigma+np.min(chi)))
    return qrr[np.argsort(dis)[:2]],chi[np.argsort(dis)[:2]]    

# Punto x e y de la media, angulo de inclinacion, eje menor y eje mayor
def para_elipsoide(puntos):
    X,Y = puntos[:,0],puntos[:,1]
    xm,ym = np.mean(X), np.mean(Y)
    d = ((X - xm)**2 + (Y - ym)**2)**0.5
    cov = np.cov(puntos.T)
    lmb, v = np.linalg.eig(cov)
    theta = np.degrees(np.arctan2(*v[:, 0][::-1].real))
    return xm,ym,theta,np.min(d)*2,np.max(d)*2 

################################################################################################################################
'Funciones minimizadoras'
################################################################################################################################
#Minimizador de la variable x
def minimizador1dx(qy0,qx,qy,N2q,N2qnq,er,pos):
    # Valor inicial y valores de y, lista de x, N_eventos bean,K,error,poisson 
    chimin = []
    for i in qx:
        chi = chi2mini(i,qy,N2q,N2qnq,er,pos)
        chimin.append(np.min(chi)) 
    return np.array(chimin) - np.min(np.array(chimin))    

#Minimizador de la variable y
def minimizador1dy(qx0,qx,qy,N2q,N2qnq,er,pos):
    # Valor inicial x,valores de x,lista de y, N_eventos bean, K, error,poisson 
    chimin = []
    for i in qy:
        chi = chi2mini(qx,i,N2q,N2qnq,er,pos)
        chimin.append(np.min(chi))  
    return np.array(chimin) - np.min(np.array(chimin))  

################################################################################################################################
'Funcion que hace que los bins sean mas realistas'
################################################################################################################################
# Hace un corte en funcion de bincorte
def cortePE (tm):
    Tcoef = np.linspace(0, tm, 1000)
    PEcoef = QFT(Tcoef,LY)  
    splines = CubicSpline(PEcoef, Tcoef)
    PEarray = np.arange(bin_corte, 61)
    Tarray = splines(PEarray)
    return Tarray

# Montecarlo de decision de aumentar y que sean almenos 1 evento
def Nrealista(Nfloat,cut):
    n = len(Nfloat)
    N,pos,lost,N_acum = [],[],0,0
    rate = np.random.uniform(size=len(Nfloat))
    for i in range(n-1,-1,-1):
        N_dec,N_int = np.modf(Nfloat[i])
        if N_dec >= rate[i] and N_int > (cut-1):
            N.append(int(N_int + 1))
            lost -= N_dec
        else :
            if N_int < cut:
                N_acum += N_dec
                if N_acum > cut:
                    N_dec,N_int = np.modf(N_acum)
                    if N_dec >= rate[i] :
                        N.append(int(N_int + 1))
                        lost -= N_dec
                    else : 
                        lost += N_dec
                        N.append(int(N_int))
                else:
                    pos.append(i)
            else:
                lost += N_dec
                N.append(int(N_int))
    return np.array(N[::-1]).astype(int),lost,np.array(pos[::-1]).astype(int)

# Distribución de eventos deacuerdo a una distribución de poisson
def Nrealistapos(Nfloat,cut):
    n = len(Nfloat)
    Np = np.random.poisson(Nfloat)
    pos = []
    Ncum = 0 
    if cut == False:
        lost = np.sum(-Nfloat+Np)
        Np = np.maximum(Np, 1)
        return Np.astype(int),lost,np.array(pos[::-1]).astype(int)
    else:
        N = []
        for i in range(n-1,-1,-1): 
            if Np[i] <= cut : 
                if Ncum == 0 : 
                    pos.append(i)
                    Ncum += Nfloat[i]
                else:
                    Ncum += Nfloat[i]
                    Nw = np.random.poisson(Ncum)
                    if Nw >= cut : 
                        Ncum = 0 
                        N.append(Nw)
                    else:
                        pos.append(i)
            else:
                N.append(Np[i])
        Np = np.array(N[::-1]).astype(int)
        lost = np.sum(Np)-np.sum(Nfloat)
        return np.array(N[::-1]).astype(int),lost,np.array(pos[::-1]).astype(int)    
        
# Hace que los beans tengan el tamaño adecuado 
def fix_bins(arr,p):
    N = []
    j = 0
    for i in range(len(arr)):
        if i == int(p[j]):
            N[-1] = np.concatenate( ( N[-1] , arr[i] ))
            j += 1
        else: 
            N.append(arr[i])
    return N      

# Genera de forma adecuada los coeficientes para los bins bidimensionales 
def temporizador_bins(N,GP,GD):
    k1 = np.tensordot(GD,N[1][bin_corte:],axes=0) + np.tensordot(GP,N[0][bin_corte:],axes=0)
    k2 = np.tensordot(GD,N[2][bin_corte:],axes=0)
    return np.array([k1,k2])    
################################################################################################################################
'Funcion que calcula los coeficientes de la convinacion lineal'
################################################################################################################################
# Proyecta la elipse en las direcciones para hacer las lineas             
def proyecion(qx,pelipse,corta):
    rad,des = np.radians(pelipse[2]+180),np.pi/2
    c1 = np.cos(rad)*pelipse[0] + np.sin(rad)*pelipse[1] 
    c2 = np.sin(rad+des)*pelipse[1] + np.cos(rad+des)*pelipse[0] 
    if corta: 
        return (c2-np.cos(rad+des)*qx)/np.sin(rad+des)
    else: 
        return (c1-np.cos(rad)*qx)/np.sin(rad)
    return None 

# Proyecta los arrays de puntos
def proyectar(qx,qy,m):
    a = 1/(1+m**2)**0.5
    b = -m*a
    xprime = a*qy + b*qx 
    yprime = b*qy - a*qx 
    coor = np.array([xprime,yprime])
    return coor  

# Calcula valores de covarianza 
def cova(puntos):
    cov = np.cov(puntos.T)
    cov = np.linalg.inv(cov)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    tm = eigenvectors
    diagonalized_cov_matrix = np.dot(tm.T, np.dot(cov, tm))
    theta_radians = np.arctan2(tm[1, 0], tm[0, 0])
    return cov,diagonalized_cov_matrix,tm,theta_radians   

################################################################################################################################
'Funciones para almacenar y leer datos e imagenes'
################################################################################################################################
# Guarda los datos de chi2 para usarlos en graftopo
def guardar_datos_Z(Z_g, Z_p, sig, nb, ffu, ffe, gff):
    factor = 'conff' if gff else 'sinff'
    nombre_archivo = f'chi{factor}_{sig}_{nb}({ffu},{ffe}).txt'
    carpeta = 'datos_elipses'
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
    ruta_completa = os.path.join(carpeta, nombre_archivo)
    with open(ruta_completa, 'w') as archivo:
        archivo.write("Datos de Z_sqr:\n")
        for array in Z_g:
            archivo.write(" ".join(map(str, array)) + "\n")
        archivo.write("\nDatos de Z_pos:\n")
        for array in Z_p:
            archivo.write(" ".join(map(str, array)) + "\n")
    return None 

# Guarda arrays en la carpeta correspondiente 
def guardar_lista_en_carpeta(lista, nombre_lista,nombre_carpeta):
    if not os.path.exists(nombre_carpeta):
        os.makedirs(nombre_carpeta)
    ruta_archivo = os.path.join(nombre_carpeta, f'{nombre_lista}.npy')
    np.save(ruta_archivo, lista)
    return None    

# Lista las carpetas de un directorio 
def listar_archivos_en_carpeta(carpeta):
    if not os.path.exists(carpeta):
        print(f'La carpeta {carpeta} no existe.')
        return True

    archivos = os.listdir(carpeta)
    archivos = [archivo for archivo in archivos if os.path.isfile(os.path.join(carpeta, archivo))]
    return archivos

# Lee las listas de numpy que hay en una carpeta con un nombre 
def leer_archivos_npy(carpeta, archivos):
    contenido_archivos = {}
    for archivo in archivos:
        ruta_archivo = os.path.join(carpeta, archivo)
        try:
            contenido = np.load(ruta_archivo, allow_pickle=True)
            contenido_archivos[archivo] = contenido
        except Exception as e:
            print(f'Error al leer el archivo {archivo}: {e}')
    return contenido_archivos  

# Lee los datos de chi2 en una carpeta
def leer_datos_de_carpeta(carpeta):
    Z_g_list,Z_p_list,FF_v,bns = [],[],[],[]
    archivos = os.listdir(carpeta)
    for archivo in sorted(archivos):
        if archivo.endswith('.txt'):  # Leer solo archivos de texto
            ruta_archivo = os.path.join(carpeta, archivo)
            Z_g_temp,Z_p_temp = [],[]
            leyendo_Z_g = True
            with open(ruta_archivo, 'r') as archivo:
                for linea in archivo:
                    linea = linea.strip()
                    if linea == "Datos de Z_sqr:":
                        leyendo_Z_g = True
                        continue
                    elif linea == "Datos de Z_pos:":
                        leyendo_Z_g = False
                        continue
                    if linea: 
                        valores = np.array(list(map(float, linea.split())))
                        if leyendo_Z_g:
                            Z_g_temp.append(valores)
                        else:
                            Z_p_temp.append(valores)
            Z_g_list.append(Z_g_temp)
            Z_p_list.append(Z_p_temp)
            v_ff = ruta_archivo.split('\\')
            v_ff = v_ff[1].split('_')
            bns.append(v_ff[2].split('(')[0])
            if v_ff[0] == 'chisinff':
                FF_v.append(False)
            else :
                FF_v.append(True)
    return Z_g_list, Z_p_list,FF_v,bns


#################################################################################################################################################
#################################################################################################################################################
'Calculos'
#################################################################################################################################################
#################################################################################################################################################

#################################################################################################################################################
"Calculos de variables comunes "
#################################################################################################################################################

'Calculos de las cargas'
Q_2_SM = Q(Z, N)                          # Carga del cesio segun el SM                       

'Calculos de variables comunes al problema'
Nt = m_det/uma_kg(m_csi_uma(Z, N))        # Numero de atomos objetivo
m_n = uma_MeV(m_csi_uma(Z, N))            # Masa de nucleo de CsI en MeV
N_mu_muon = n_pot * f_nu_p                # Numero de neutrinos muonicos
L = m_to_Mev(L_m)                         # Longitud en MeV
ns_mev = m_to_Mev(ns)                     # Nuclear skin en Mev 
R_p_mev = m_to_Mev(R_p)                   # Radio del proton en Mev
R_n_mev = m_to_Mev(R_n)                   # Radio del neutron en Mev 

'Calculo de las energias maximas y mínimas'
E_max = m_mu/2                            # Maximo  de energia en el delay
tmax_prompt = 2*E_nu**2/(m_n + 2*E_nu)    # Maximo de energia T en el prompt 
tmax_delay = m_mu**2/(2*(m_mu + m_n))     # Maximo de energia T en el delayed
t_max = np.max([tmax_delay,tmax_prompt])  # Elegimos un t_max comun 
t_min = 7.9*1e-9                          # Valor minimo para evitar infinitos

#################################################################################################################################################
'Ajustes de array segun requerimiento'
#################################################################################################################################################

'Calculamos la funcion inversa del Quenchin Factor'
T_coef = np.linspace(t_min, t_max, 1000)
PE_coef = QFT(T_coef,LY)  
inverse_QFT_spline = CubicSpline(PE_coef, T_coef)        
        
"Generamos una array de los valores de T que vamos a tener"  
if origenT : 
    t_array = np.linspace(t_min,tmax_delay,t_split)    
    PE_array = QFT(t_array,LY) 
    
else : 
    'Calculo el array de PE y de T'
    PE_array = np.linspace(min(PE_coef), max(PE_coef), t_split)
    t_array = inverse_QFT_spline(PE_array)
    
'Introducimos el corte de PE en el caso de ser necesario'
if cortarPE:
    num_bins = 60 - bin_corte 
    rangosT = cortePE(t_max)
    t_array,t_bins = [],[]
    
    for i in range(len(rangosT) - 1):
        lsi=np.linspace(rangosT[i],rangosT[i+1],num_split_bins,endpoint=False)
        t_array.extend(lsi)
        t_bins.append(lsi)
    t_bins = np.array(t_bins)
    t_array = np.array(t_array)
    
    if origenT:
        PE_array = QFT(t_array,LY)       # Para usarlo donde las eficiciencias
else: 
    t_bins = np.array_split(t_array, num_bins)          
    

#################################################################################################################################################
'Calculos de DN/DT, DN/DPE y el numero de eventos'
#################################################################################################################################################

"Calculamos puntos para realizar la gráfica de dN/DT"
dn_list,dn_promt_list,dn_delay_e_list,dn_delay_u_list = dn_calculator(t_array,Z,N,m_det,E_nu,E_max,n_pot,f_nu_p,Q_2_SM,m_mu,L,True,ff)
dn_listnq,dn_promt_listnq,dn_delay_e_listnq,dn_delay_u_listnq = dn_calculator(t_array,Z,N,m_det,E_nu,E_max,n_pot,f_nu_p,Q_2_SM,m_mu,L,False,ff)
dn_listq2,dn_promt_listq2,dn_delay_e_listq2,dn_delay_u_listq2 = dn_calculator2q(t_array,Z,N,m_det,E_nu,E_max,n_pot,f_nu_p,m_mu,L,ff)

'Calculamos el Jacobiano en Mev para pasar de T a PE'
inverse_QFT_k_spline_MeV = CubicSpline(QFT(t_array,LY), t_array)
jacobian_QFT_k_in = inverse_QFT_k_spline_MeV.derivative()
J_T_PE_Mev = jacobian_QFT_k_in(PE_array)

"Calculamos puntos para realizar la gráfica de dN/DPE"
dn_list_pe,dn_promt_list_pe,dn_delay_e_list_pe,dn_delay_u_list_pe = J_T_PE_Mev*dn_list,J_T_PE_Mev*dn_promt_list,J_T_PE_Mev*dn_delay_e_list,J_T_PE_Mev*dn_delay_u_list
dn_listnq_pe,dn_delay_u_listnq_pe,dn_delay_e_listnq_pe,dn_promt_listnq_pe = J_T_PE_Mev*dn_listnq,J_T_PE_Mev*dn_promt_listnq,J_T_PE_Mev*dn_delay_e_listnq,J_T_PE_Mev*dn_delay_u_listnq


#################################################################################################################################################
'Calculos del flujo '
#################################################################################################################################################

if grafflux : 
    Ener = np.linspace(0,E_max,10000)
    E_ind = np.argmin(np.abs(Ener-E_nu)) 
    flux_nu = dphi_mu_dE_nu(n_pot,f_nu_p,E_nu,L)/20
    impulse_signal = signal.unit_impulse(len(Ener),E_ind)*flux_nu
    flux_ve = dphi_ve_dE(Ener,n_pot,f_nu_p,m_mu,L)
    flux_vu = dphi_vmu_dE(Ener,n_pot,f_nu_p,m_mu,L)  

#################################################################################################################################################
'Calculos del flujo '
#################################################################################################################################################
"""Esta parte calcula la gráfica del flujo temporal, pero es recomendable  
calcular con anterioridad grafmasterdata, si no se calcula aquí"""

if graf_time : 
    t_data_list = listar_archivos_en_carpeta('t_data')
    if t_data_list or num_bins_temp != 24: 
        'Defino los array temporales'
        timepass = 10000                          # Pongo un timepass normalito
        num_bins_temp = int(2*(timef-time0))      # Ajusto a intervalos de 0.5
        time_arr = np.linspace(time0,timef,timepass)
        time_bins = np.array_split(time_arr, num_bins_temp)

        # Calculamos el flujo de neutrinos temporal. 
        DP_dtM = dP_dt_mu(time_arr,tau_mu,tau_pi) 
        P_mu_splin = CubicSpline(time_arr, DP_dtM)  # Aproximo P_mu por splines
        DP_dtE = dP_dt_ee(time_arr,tau_mu,tau_pi)
        
        'Aplicamos la eficiencia temporal'
        if T_efici: 
            E_time = eft(time_arr)
            DP_dtM = DP_dtM*E_time
            DP_dtE = DP_dtE*E_time
            
        'Calculamos los valores de g_j'
        DP_dtM_bins = np.array_split(DP_dtM, num_bins_temp)
        DP_dtE_bins = np.array_split(DP_dtE, num_bins_temp)
        g_p = split_bins(num_bins_temp, DP_dtM_bins, time_bins)
        g_d = split_bins(num_bins_temp, DP_dtE_bins, time_bins)
        DP_dtM_norm,g_p =  DP_dtM/ np.sum(g_p),g_p/np.sum(g_p)
        DP_dtE_norm,g_d =  DP_dtE/ np.sum(g_d),g_d/np.sum(g_d)
        
    else: 
        'Los saco directamente de los calculos de grafmasterdata'
        DP_dtM_norm,g_p = t_data_list
        
#################################################################################################################################################
'Calculos del número de eventos de una forma naive'
#################################################################################################################################################
'Realizo una integración y el número de eventos con la integral naive'
if N_eventos_naive :
    N_prompt = simpson(y=dn_promt_list, x = t_array)
    N_delay_u = simpson(y=dn_delay_u_list,x = t_array)
    N_delay_e = simpson(y=dn_delay_e_list, x = t_array)
    N_tot = N_prompt + N_delay_e + N_delay_u

    N_promptq2 = simpson(y=dn_promt_listq2, x = t_array)
    N_delay_uq2 = simpson(y=dn_delay_u_listq2,x = t_array)
    N_delay_eq2 = simpson(y=dn_delay_e_listq2, x = t_array)
    N_totq2 = N_promptq2 + N_delay_eq2 + N_delay_uq2

#################################################################################################################################################
'Calculos de las graficas de DN/DT y DN/DPE'
#################################################################################################################################################

if grafTee or graf2q_Tee:
    if cortarPE: 
        PE_array = np.linspace(bin_corte, max(PE_coef), t_split)
        t_array_qf = inverse_QFT_spline(PE_array)
    else: 
        t_array_qf = t_array
        PE_array = QFT(t_array_qf,LY)
        
    if graf2q_Tee: 
        dn_list_pe,dn_promt_list_pe,dn_delay_e_list_pe,dn_delay_u_list_pe = dn_calculator(t_array_qf,Z,N,m_det,E_nu,E_max,n_pot,f_nu_p,Q_2_SM,m_mu,L,True,ff)
        dn_list2q_pe,dn_promt_list2q_pe,dn_delay_e_list2q_pe,dn_delay_u_list2q_pe = dn_calculator2q(t_array_qf,Z,N,m_det,E_nu,E_max,n_pot,f_nu_p,m_mu,L,ff)
        dn_listnq_pe,dn_delay_u_listnq_pe,dn_delay_e_listnq_pe,dn_promt_listnq_pe = dn_calculator(t_array_qf,Z,N,m_det,E_nu,E_max,n_pot,f_nu_p,Q_2_SM,m_mu,L,False,ff)
        
        # Definimos el jacobiano 
        JA2 = jacobian_QFT_k_in(PE_array)
        dn_list_pe,dn_promt_list_pe,dn_delay_e_list_pe,dn_delay_u_list_pe = JA2*dn_list_pe,JA2*dn_promt_list_pe,JA2*dn_delay_e_list_pe,JA2*dn_delay_u_list_pe
        dn_list2q_pe,dn_promt_list2q_pe,dn_delay_e_list2q_pe,dn_delay_u_list2q_pe = JA2*dn_list2q_pe,JA2*dn_promt_list2q_pe,JA2*dn_delay_e_list2q_pe,JA2*dn_delay_u_list2q_pe 
        
#################################################################################################################################################
'Calculos del las funciones chi^2 en 1 dimensión'
#################################################################################################################################################

if grafchi1:
    'Separamos todo de forma adecuada'
    dn_list_bins = np.array_split(dn_list, num_bins)  #Array de puntos de dN/DT
    dn_listnq_bins = np.array_split(dn_listnq, num_bins) 
    N_bins_teo = split_bins(num_bins,dn_list_bins,t_bins)

    'Decidimos si queremos unos valores realistas o no y si juntar bins'
    if real :
        N_bins,lost,pos = Nrealistapos(N_bins_teo,corte) 
        
    else: 
        N_bins,lost,pos = N_bins_teo,0,[]

    if num_bins != len(N_bins) :
        t_bins = fix_bins(t_bins,pos)
        dn_listnq_bins = fix_bins(dn_listnq_bins,pos)
        num_bins_new = len(N_bins)
        N_binsnq = split_bins(num_bins_new,dn_listnq_bins,t_bins)       
            
    else:
        N_binsnq = split_bins(num_bins,dn_listnq_bins,t_bins)    
        
        
    'Calculamos las cosas restantes para el calculo de chi de 1 carga'
    extremos = np.array([Q_2_SM*(1-f_e),Q_2_SM*(1+f_e)])          
    q_array = np.linspace(extremos[0], extremos[1],split)       
        
    Nred,Kred = np.stack([N_bins] * split),np.stack([N_binsnq] * split)
    qred = np.stack([q_array] * len(N_bins), axis=1)
    Errorred_sqr = [N_bins**0.5]*split
         
    'Calculamos la funcion chi'
    chi_q_sqrI = chi_red(qred,Nred,Kred,Errorred_sqr,False)
    chi_q_pos = chi_red(qred,Nred,Kred,None,True)
        
    'Calculamos donde tengo el valor de 1 sigma'
    Q_2_sigma_sqr,chi_q_sigma_sqr = puntossigmachi(chi_q_sqrI,q_array)
    Q_2_sigma_pos,chi_q_sigma_pos = puntossigmachi(chi_q_pos,q_array)

#################################################################################################################################################
'Calculos del las funciones chi^2 en 2 dimensiones para primera aproximación '
#################################################################################################################################################

if grafchi2q or grafchi2q_3d or grafmini1d or graffcorr or graftopo :
    # Separamos todo de forma adecuada
    dn_list_bins = np.array_split(dn_list, num_bins)  #Array de puntos de dN/DT
    dn_listnq_bins = np.array_split(dn_listnq, num_bins) 
    N_bins_teo = split_bins(num_bins,dn_list_bins,t_bins)
    
    'Decidimos si queremos unos valores realistas o no y si juntar bins'
    if real :
        N_bins,lost,pos = Nrealistapos(N_bins_teo,corte) 
        
    else: 
        N_bins,lost,pos = N_bins_teo,0,[]
        
    'Calculamos los coeficientes para el chi^2' 
    dn_list_u,dn_list_e = dn_delay_u_listnq + dn_promt_listnq, dn_delay_e_listnq 
    Kq1,Kq2 = np.array_split(dn_list_u, num_bins),np.array_split(dn_list_e, num_bins)
        
    if num_bins != len(N_bins) :
        t_bins = fix_bins(t_bins,pos)
        Kq1 = fix_bins(Kq1,pos)
        Kq2= fix_bins(Kq2,pos)
          
    N_binsnq2q =np.array([split_bins(len(Kq1),Kq1,t_bins),split_bins(len(Kq1),Kq2,t_bins)])

    'Generamos el array de Q_mu y Q_e'
    extremos_mu = np.array([Q_2_SM*(1-fu),Q_2_SM*(1+fu)])     
    extremos_ee = np.array([Q_2_SM*(1-fe),Q_2_SM*(1+fe)])     
    q_array_mu = np.linspace(extremos_mu[0], extremos_mu[1],splitred) 
    q_array_ee = np.linspace(extremos_ee[0], extremos_ee[1],splitred)
    q_red = np.vstack((q_array_mu, q_array_ee))
    q_meshgrid = np.meshgrid(*q_red)
    qu,qe = q_meshgrid
    
    'Calculamos la funciones chi^2, junto con sus mínimos"'
    if grafchi2q or grafchi2q_3d or graffcorr:
        chi_q_sqr_2q = chi_2q(q_meshgrid,N_bins,N_binsnq2q,N_bins**0.5,False)
        chi_q_pos_2q = chi_2q(q_meshgrid,N_bins,N_binsnq2q,None,True) 
        Z_sqr= (chi_q_sqr_2q - np.min(chi_q_sqr_2q))
        Z_pos= (chi_q_pos_2q - np.min(chi_q_pos_2q))    
        x_min_sqr, y_min_sqr = np.unravel_index(np.argmin(chi_q_sqr_2q),chi_q_sqr_2q.shape)
        x_min_pos, y_min_pos = np.unravel_index(np.argmin(chi_q_pos_2q),chi_q_pos_2q.shape)
        
        if saveZ: # Guarda los datos para usarlos en graftopo
            guardar_datos_Z(Z_sqr, Z_pos, sigmalegend,len(N_bins),fu, fe, ff)
    
    if grafmini1d:
        'Minimizados qe y variamos qu'
        chix_sqr = minimizador1dx(Q_2_SM,q_array_mu,q_array_ee,N_bins,N_binsnq2q,N_bins**0.5,False)
        chix_pos = minimizador1dx(Q_2_SM,q_array_mu,q_array_ee,N_bins,N_binsnq2q,None,True)
            
        'Minimizados qu y variamos qe'
        chiy_sqr = minimizador1dy(Q_2_SM,q_array_mu,q_array_ee,N_bins,N_binsnq2q,N_bins**0.5,False)
        chiy_pos = minimizador1dy(Q_2_SM,q_array_mu,q_array_ee,N_bins,N_binsnq2q,None,True)
            
        'Calculamos los puntos a 1 sigma'
        Q_2_sigma_sqr_2qu,chi_q_sigma_sqr_2qu = puntossigmachi(chix_sqr,q_array_mu)
        Q_2_sigma_pos_2qu,chi_q_sigma_pos_2qu = puntossigmachi(chix_pos,q_array_mu)
        Q_2_sigma_sqr_2qe,chi_q_sigma_sqr_2qe = puntossigmachi(chiy_sqr,q_array_ee)
        Q_2_sigma_pos_2qe,chi_q_sigma_pos_2qe = puntossigmachi(chiy_pos,q_array_ee)

#################################################################################################################################################
'Calculos del las correlaciones para primera aproximación'
#################################################################################################################################################

if graffcorr:
    'Encontramos los puntos dentro de la elipse'
    sqr_indices = np.where(chi_q_sqr_2q <= sigma)  
    pos_indices = np.where(chi_q_pos_2q <= sigma)
    elipsequ_sqr = np.column_stack((qu[sqr_indices], qe[sqr_indices]))
    elipsequ_pos = np.column_stack((qu[pos_indices], qe[pos_indices]))
    
    'Calculamos los ejes de la elipse'
    param_elips_sqr = para_elipsoide(elipsequ_sqr)
    param_elips_sqr = para_elipsoide(elipsequ_pos)
    
    indeterminado = proyecion(q_array_mu,param_elips_sqr,False) # con -
    determinado = proyecion(q_array_mu,param_elips_sqr,True)    # con +
    
    'Calculo las rectas y los coeficientes de la combinacion lineal'
    mi, bi = np.polyfit(q_array_mu, indeterminado,1) 
    md, bd = np.polyfit(q_array_mu, determinado,1) 
    a_coef = 1/(1+mi**2)**0.5
    b_coef = -mi*a_coef
    
    'Proyecciones para hacer la grafica'
    elip_cor_sqr = proyectar(qu[sqr_indices],qe[sqr_indices],mi)
    elip_cor_pos = proyectar(qu[pos_indices],qe[pos_indices],mi)
    eje_det = proyectar(q_array_mu,indeterminado,mi)
    eje_indet = proyectar(q_array_mu,determinado,mi)
    Q_2_SM_cx,Q_2_SM_cy = proyectar(Q_2_SM,Q_2_SM,mi)
    y_min_pos_c,x_min_pos_c = proyectar(x_min_pos,y_min_pos,mi)
    y_min_sqr_c,x_min_sqr_c = proyectar(x_min_sqr,y_min_sqr,mi)
    
    'Calculos de correlacion matematica y covarianza'
    cov_sqr,cov_diag_sqr,trans_sqr,ang_sqr = cova(elipsequ_sqr)
    cov_pos,cov_diag_pos,trans_pos,ang_pos = cova(elipsequ_pos)
    std_devs = np.sqrt(np.diag(cov_sqr))
    std_matrix = np.outer(std_devs, std_devs)
    corr_matrix = cov_sqr/std_matrix
    
    cov_sqr_C,cov_diag_sqr_C,trans_sqr_C,ang_sqr_C = cova(elip_cor_sqr.T)
    std_devs_C = np.sqrt(np.diag(cov_sqr_C))
    std_matrix_C = np.outer(std_devs_C, std_devs_C)
    corr_matrix_C = cov_sqr_C/std_matrix_C
        
    
#################################################################################################################################################
'Calculos de la forma del factor de forma'
#################################################################################################################################################
# Factor de forma nuclear 
if graf_FF: 
    mn = uma_MeV(m_csi_uma(Z, N))
    q_2_ar = (t_array*2*mn)**0.5
    F_p,qrop = Factos_nucleon(q_2_ar,ns_mev,R_p_mev)
    F_n,qron = Factos_nucleon(q_2_ar,ns_mev,R_n_mev)
    #F_0 = Factos_nucleon(q(t_array[-1]),ns_mev,R_n_mev)
    F_tot = 1/((N-Z) - 0.076*Z)*((N-Z)*F_n - 0.076*Z*F_p)

#################################################################################################################################################
'Calculos con la inclusion de consideraciones experimentales'
#################################################################################################################################################

if grafJacob or grafefici or grafresol or grafmasterdatqa :
    
    'Definimos correctamete los arrays, sin importar el cortePE o OrigenT'
    num_bins = 60 
    
    'Generamos el array de PE'
    PE_array_rec = np.linspace(min(PE_coef), num_bins, t_split_rec)
    
    'Generamos los array en MeV'
    tee_array_rec = PE_array_rec /LY
    t_array_rec = inverse_QFT_spline(PE_array_rec) 
    
    'Corregimos los valores de los array a KeV'
    t_array_rec_k = t_array_rec*1e3
    tee_array_rec_k = tee_array_rec*1e3
    
    'Calculamos las funciones inversas y sus derivadas'
    inverse_QF_k_spline = CubicSpline(QF_k(t_array_rec_k), t_array_rec_k)     # Evaluar sobre tee_array_rec_k
    inverse_QFT_k_spline = CubicSpline(QFT_k(t_array_rec_k), t_array_rec_k)   # Evaluar sobre PE_array_rec
    jacobian_QF_k_in = inverse_QF_k_spline.derivative()                 # Evaluar sobre tee_array_rec_k
    jacobian_QFT_k_in = inverse_QFT_k_spline.derivative()                 # Evaluar sobre PE_array_rec

    'Obtenemos los 4 jacobianos'
    J_T_PE = D_QFT_k(t_array_rec_k)                # Jacobiano de PE a T 
    J_PE_T = jacobian_QFT_k_in(PE_array_rec)       # Jacobiano de T a PE 
    J_T_TE = D_QF_k(t_array_rec_k)                 # Jacobiano de Te a T 
    J_TE_T = jacobian_QF_k_in(tee_array_rec_k)     # Jacobiano de T a Te

    if grafresol or grafmasterdatqa or grafefici:
        'Generamos el Grild'
        PE_rec_grid, Tee_grid = np.meshgrid(PE_array_rec, tee_array_rec_k)
        PE_bins_rec = np.array_split(PE_array_rec, num_bins)
        t_bins_rec = np.array_split(t_array_rec_k, num_bins)
        Pe_range = np.arange(0,60,1) 
        
        'Genero los array temporales'
        time_arrchi = np.linspace(time0,timef,timepass)
        time_binschi = np.array_split(time_arrchi, num_bins_temp)
        
        'Calculamos la matriz R y la E'
        R_matrix = energy_resolution_origin(PE_rec_grid, Tee_grid)
        R_matrix = np.multiply(R_matrix,J_TE_T)                  
        E_pe = efpe(PE_array_rec)
        R_diag = np.eye((len(t_array_rec))) 
        R_diag = np.multiply(R_diag,J_T_TE) 
        E_one = np.ones((len(t_array_rec)))

        'Calculamos algunos cortes para las graficas'
        if grafresol:
            index40 = np.argmin(np.abs(PE_array_rec - 40))
            index20 = np.argmin(np.abs(PE_array_rec - 20))
            index30 = np.argmin(np.abs(PE_array_rec - 30))
            index15 = np.argmin(np.abs(PE_array_rec - 15))
            R_40 = R_matrix[index40][:]
            R_20 = R_matrix[index20][:]
            R_30 = R_matrix[index30][:]
            R_15 = R_matrix[index15][:]

    if grafmasterdatqa: 
        "Calculamos puntos para realizar la gráfica de dN/DT"
        dn_list_rec,dn_p_rec,dn_de_rec,dn_du_rec = dn_calculator(t_array_rec,Z,N,m_det,E_nu,E_max,n_pot,f_nu_p,Q_2_SM,m_mu,L,True,True)
        dn_list_recnq,dn_p_recnq,dn_de_recnq,dn_du_recnq = dn_calculator(t_array_rec,Z,N,m_det,E_nu,E_max,n_pot,f_nu_p,Q_2_SM,m_mu,L,False,True)
        dn_u_recnq = dn_p_recnq + dn_du_recnq
        dn_d_rec,dn_e_recnq = dn_du_rec + dn_de_rec, dn_de_recnq 
        
        "Calculamos los puntos para la gradica dN/DT sin el factor de forma"
        dn_list_rec_f,dn_p_rec_f,dn_de_rec_f,dn_du_rec_f = dn_calculator(t_array_rec,Z,N,m_det,E_nu,E_max,n_pot,f_nu_p,Q_2_SM,m_mu,L,True,False)
        dn_list_recnq_f,dn_p_recnq_f,dn_de_recnq_f,dn_du_recnq_f = dn_calculator(t_array_rec,Z,N,m_det,E_nu,E_max,n_pot,f_nu_p,Q_2_SM,m_mu,L,False,False)
        dn_u_recnq_f = dn_p_recnq_f + dn_du_recnq_f
        dn_d_rec_f,dn_e_recnq_f = dn_du_rec_f + dn_de_rec_f, dn_de_recnq_f 

        'Parte temporal' 
        # Flujo temporal de neutrinos prompt         
        DP_dtMchi = dP_dt_mu(time_arrchi,tau_mu,tau_pi) 
        P_mu_splin = CubicSpline(time_arrchi, DP_dtMchi)
        DP_dtEchi = dP_dt_ee(time_arrchi,tau_mu,tau_pi)
    
        # Normalizo los valores 
        E_time = eft(time_arrchi) 
        DP_dtMchi = DP_dtMchi*E_time
        DP_dtEchi = DP_dtEchi*E_time
            
        # Calculamos los valores de g_j
        DP_dtM_binschi = np.array_split(DP_dtMchi, num_bins_temp)
        DP_dtE_binschi = np.array_split(DP_dtEchi, num_bins_temp)
        g_pchi = split_bins(num_bins_temp, DP_dtM_binschi, time_binschi)
        g_dchi = split_bins(num_bins_temp, DP_dtE_binschi, time_binschi)
        
        # Normalizamos los valores 
        DP_dtM_norm,g_p =  DP_dtMchi/ np.sum(g_pchi),g_pchi/np.sum(g_pchi)
        DP_dtE_norm,g_d =  DP_dtEchi/ np.sum(g_dchi),g_dchi/np.sum(g_dchi)  

        'Calculamos los numero de eventos en cada caso con factor de forma'
        N_bins_p_rec = doble_integral(dn_p_rec,PE_bins_rec,t_array_rec_k,R_diag,E_one,num_bins)
        N_bins_p_RE = doble_integral(dn_p_rec,PE_bins_rec,t_array_rec_k,R_matrix,E_pe,num_bins)
        N_bins_p_R = doble_integral(dn_p_rec,PE_bins_rec,t_array_rec_k,R_matrix,E_one,num_bins)
        N_bins_p_E = doble_integral(dn_p_rec,PE_bins_rec,t_array_rec_k,R_diag,E_pe,num_bins)
        
        N_bins_de_rec = doble_integral(dn_de_rec,PE_bins_rec,t_array_rec_k,R_diag,E_one,num_bins)
        N_bins_de_RE = doble_integral(dn_de_rec,PE_bins_rec,t_array_rec_k,R_matrix,E_pe,num_bins)
        N_bins_de_R = doble_integral(dn_de_rec,PE_bins_rec,t_array_rec_k,R_matrix,E_one,num_bins)
        N_bins_de_E = doble_integral(dn_de_rec,PE_bins_rec,t_array_rec_k,R_diag,E_pe,num_bins)
        
        N_bins_du_rec = doble_integral(dn_du_rec,PE_bins_rec,t_array_rec_k,R_diag,E_one,num_bins)
        N_bins_du_RE = doble_integral(dn_du_rec,PE_bins_rec,t_array_rec_k,R_matrix,E_pe,num_bins)
        N_bins_du_R = doble_integral(dn_du_rec,PE_bins_rec,t_array_rec_k,R_matrix,E_one,num_bins)
        N_bins_du_E = doble_integral(dn_du_rec,PE_bins_rec,t_array_rec_k,R_diag,E_pe,num_bins)

        N_bins_rec = N_bins_p_rec + N_bins_de_rec + N_bins_du_rec 
        N_bins_RE = N_bins_p_RE + N_bins_de_RE + N_bins_du_RE 
        N_bins_E = N_bins_p_E + N_bins_de_E + N_bins_du_E 
        N_bins_R = N_bins_p_R + N_bins_de_R + N_bins_du_R 
        
        'Calculamos los coeficientes de nq con factor de forma'
        N_binsnq_p_rec = doble_integral(dn_p_recnq,PE_bins_rec,t_array_rec_k,R_diag,E_one,num_bins)
        N_binsnq_p_RE = doble_integral(dn_p_recnq,PE_bins_rec,t_array_rec_k,R_matrix,E_pe,num_bins)
        N_binsnq_p_R = doble_integral(dn_p_recnq,PE_bins_rec,t_array_rec_k,R_matrix,E_one,num_bins)
        N_binsnq_p_E = doble_integral(dn_p_recnq,PE_bins_rec,t_array_rec_k,R_diag,E_pe,num_bins)

        N_binsnq_du_rec = doble_integral(dn_du_recnq,PE_bins_rec,t_array_rec_k,R_diag,E_one,num_bins)
        N_binsnq_du_RE = doble_integral(dn_du_recnq,PE_bins_rec,t_array_rec_k,R_matrix,E_pe,num_bins)
        N_binsnq_du_R = doble_integral(dn_du_recnq,PE_bins_rec,t_array_rec_k,R_matrix,E_one,num_bins)
        N_binsnq_du_E = doble_integral(dn_du_recnq,PE_bins_rec,t_array_rec_k,R_diag,E_pe,num_bins)
        
        N_binsnq_e_rec = doble_integral(dn_e_recnq,PE_bins_rec,t_array_rec_k,R_diag,E_one,num_bins)
        N_binsnq_e_RE = doble_integral(dn_e_recnq,PE_bins_rec,t_array_rec_k,R_matrix,E_pe,num_bins)
        N_binsnq_e_R = doble_integral(dn_e_recnq,PE_bins_rec,t_array_rec_k,R_matrix,E_one,num_bins)
        N_binsnq_e_E = doble_integral(dn_e_recnq,PE_bins_rec,t_array_rec_k,R_diag,E_pe,num_bins)
        
        N_binsnq_u_rec = N_binsnq_p_rec + N_binsnq_du_rec 
        N_binsnq_u_RE = N_binsnq_p_RE + N_binsnq_du_RE  
        N_binsnq_u_R = N_binsnq_p_R + N_binsnq_du_R 
        N_binsnq_u_E = N_binsnq_p_E + N_binsnq_du_E 
        
        'Calculamos los numero de eventos en cada caso sin factor de forma'
        N_bins_p_rec_f = doble_integral(dn_p_rec_f,PE_bins_rec,t_array_rec_k,R_diag,E_one,num_bins)
        N_bins_p_RE_f = doble_integral(dn_p_rec_f,PE_bins_rec,t_array_rec_k,R_matrix,E_pe,num_bins)
        N_bins_p_R_f = doble_integral(dn_p_rec_f,PE_bins_rec,t_array_rec_k,R_matrix,E_one,num_bins)
        N_bins_p_E_f = doble_integral(dn_p_rec_f,PE_bins_rec,t_array_rec_k,R_diag,E_pe,num_bins)
        
        N_bins_de_rec_f = doble_integral(dn_de_rec_f,PE_bins_rec,t_array_rec_k,R_diag,E_one,num_bins)
        N_bins_de_RE_f = doble_integral(dn_de_rec_f,PE_bins_rec,t_array_rec_k,R_matrix,E_pe,num_bins)
        N_bins_de_R_f = doble_integral(dn_de_rec_f,PE_bins_rec,t_array_rec_k,R_matrix,E_one,num_bins)
        N_bins_de_E_f = doble_integral(dn_de_rec_f,PE_bins_rec,t_array_rec_k,R_diag,E_pe,num_bins)
        
        N_bins_du_R_f = doble_integral(dn_du_rec_f,PE_bins_rec,t_array_rec_k,R_matrix,E_one,num_bins)
        N_bins_du_E_f = doble_integral(dn_du_rec_f,PE_bins_rec,t_array_rec_k,R_diag,E_pe,num_bins)
        N_bins_du_rec_f = doble_integral(dn_du_rec_f,PE_bins_rec,t_array_rec_k,R_diag,E_one,num_bins)
        N_bins_du_RE_f = doble_integral(dn_du_rec_f,PE_bins_rec,t_array_rec_k,R_matrix,E_pe,num_bins)
        
        N_bins_rec_f = N_bins_p_rec_f + N_bins_de_rec_f + N_bins_du_rec_f 
        N_bins_RE_f = N_bins_p_RE_f + N_bins_de_RE_f + N_bins_du_RE_f 
        N_bins_E_f = N_bins_p_E_f + N_bins_de_E_f + N_bins_du_E_f 
        N_bins_R_f = N_bins_p_R_f + N_bins_de_R_f + N_bins_du_R_f 
        
        'Calculamos los coeficientes de nq sin factor de forma'
        N_binsnq_e_rec_f = doble_integral(dn_e_recnq_f,PE_bins_rec,t_array_rec_k,R_diag,E_one,num_bins)
        N_binsnq_e_RE_f = doble_integral(dn_e_recnq_f,PE_bins_rec,t_array_rec_k,R_matrix,E_pe,num_bins)
        N_binsnq_e_R_f = doble_integral(dn_e_recnq_f,PE_bins_rec,t_array_rec_k,R_matrix,E_one,num_bins)
        N_binsnq_e_E_f = doble_integral(dn_e_recnq_f,PE_bins_rec,t_array_rec_k,R_diag,E_pe,num_bins)
        
        N_binsnq_p_rec_f = doble_integral(dn_p_recnq_f,PE_bins_rec,t_array_rec_k,R_diag,E_one,num_bins)
        N_binsnq_p_RE_f = doble_integral(dn_p_recnq_f,PE_bins_rec,t_array_rec_k,R_matrix,E_pe,num_bins)
        N_binsnq_p_R_f = doble_integral(dn_p_recnq_f,PE_bins_rec,t_array_rec_k,R_matrix,E_one,num_bins)
        N_binsnq_p_E_f = doble_integral(dn_p_recnq_f,PE_bins_rec,t_array_rec_k,R_diag,E_pe,num_bins)
        
        N_binsnq_du_rec_f = doble_integral(dn_du_recnq_f,PE_bins_rec,t_array_rec_k,R_diag,E_one,num_bins)
        N_binsnq_du_RE_f = doble_integral(dn_du_recnq_f,PE_bins_rec,t_array_rec_k,R_matrix,E_pe,num_bins)
        N_binsnq_du_R_f = doble_integral(dn_du_recnq_f,PE_bins_rec,t_array_rec_k,R_matrix,E_one,num_bins)
        N_binsnq_du_E_f = doble_integral(dn_du_recnq_f,PE_bins_rec,t_array_rec_k,R_diag,E_pe,num_bins)
        
        N_binsnq_u_rec_f = N_binsnq_p_rec_f + N_binsnq_du_rec_f
        N_binsnq_u_RE_f = N_binsnq_p_RE_f + N_binsnq_du_RE_f
        N_binsnq_u_R_f = N_binsnq_p_R_f + N_binsnq_du_R_f
        N_binsnq_u_E_f = N_binsnq_p_E_f + N_binsnq_du_E_f
    
        'Con solo una integral'
        dn_list_bins = np.array_split(dn_list, num_bins)  #Array de puntos de dN/DT
        dn_list_rec_bins = np.array_split(dn_list_rec, num_bins) 
        N_bins_una_integral = split_bins(num_bins,dn_list_rec_bins,t_bins_rec )
    
        "Guardamos los datos"
        datos_dict = {
        't_data': [(DP_dtM_norm, 'DP_dtM_norm'),(DP_dtE_norm, 'DP_dtE_norm'),(g_p, 'g_p'),(g_d, 'g_d')],
        'N_eventos': [(N_bins_p_rec, 'N_bins_p_rec'),(N_bins_p_RE, 'N_bins_p_RE'),(N_bins_p_R, 'N_bins_p_R'),(N_bins_p_E, 'N_bins_p_E'),
            (N_bins_de_rec, 'N_bins_de_rec'),(N_bins_de_RE, 'N_bins_de_RE'),(N_bins_de_R, 'N_bins_de_R'),(N_bins_de_E, 'N_bins_de_E'),
            (N_bins_du_rec, 'N_bins_du_rec'),(N_bins_du_RE, 'N_bins_du_RE'),(N_bins_du_R, 'N_bins_du_R'),(N_bins_du_E, 'N_bins_du_E'),
            (N_bins_rec, 'N_bins_rec'),(N_bins_RE, 'N_bins_RE'),(N_bins_E, 'N_bins_E'),(N_bins_R, 'N_bins_R') ],
        'N_eventosnq': [(N_binsnq_u_rec, 'N_binsnq_u_rec'),(N_binsnq_u_RE, 'N_binsnq_u_RE'),(N_binsnq_u_R, 'N_binsnq_u_R'),(N_binsnq_u_E, 'N_binsnq_u_E'),
            (N_binsnq_e_rec, 'N_binsnq_e_rec'),(N_binsnq_e_RE, 'N_binsnq_e_RE'),(N_binsnq_e_R, 'N_binsnq_e_R'),(N_binsnq_e_E, 'N_binsnq_e_E'),
            (N_binsnq_p_rec, 'N_binsnq_p_rec'),(N_binsnq_p_RE, 'N_binsnq_p_RE'),(N_binsnq_p_R, 'N_binsnq_p_R'),(N_binsnq_p_E, 'N_binsnq_p_E'),
            (N_binsnq_du_rec, 'N_binsnq_du_rec'),(N_binsnq_du_RE, 'N_binsnq_du_RE'),(N_binsnq_du_R, 'N_binsnq_du_R'),(N_binsnq_du_E, 'N_binsnq_du_E')],
        'N_eventos_f': [(N_bins_p_rec_f, 'N_bins_p_rec_f'), (N_bins_p_RE_f, 'N_bins_p_RE_f'), (N_bins_p_R_f, 'N_bins_p_R_f'), (N_bins_p_E_f, 'N_bins_p_E_f'),
                    (N_bins_de_rec_f, 'N_bins_de_rec_f'), (N_bins_de_RE_f, 'N_bins_de_RE_f'), (N_bins_de_R_f, 'N_bins_de_R_f'), (N_bins_de_E_f, 'N_bins_de_E_f'),
                    (N_bins_du_rec_f, 'N_bins_du_rec_f'), (N_bins_du_RE_f, 'N_bins_du_RE_f'), (N_bins_du_R_f, 'N_bins_du_R_f'), (N_bins_du_E_f, 'N_bins_du_E_f'),
                    (N_bins_rec_f, 'N_bins_rec_f'), (N_bins_RE_f, 'N_bins_RE_f'), (N_bins_E_f, 'N_bins_E_f'), (N_bins_R_f, 'N_bins_R_f')],
        'N_eventosnq_f': [(N_binsnq_u_rec_f, 'N_binsnq_u_rec_f'), (N_binsnq_u_RE_f, 'N_binsnq_u_RE_f'), (N_binsnq_u_R_f, 'N_binsnq_u_R_f'), (N_binsnq_u_E_f, 'N_binsnq_u_E_f'),
                      (N_binsnq_e_rec_f, 'N_binsnq_e_rec_f'), (N_binsnq_e_RE_f, 'N_binsnq_e_RE_f'), (N_binsnq_e_R_f, 'N_binsnq_e_R_f'), (N_binsnq_e_E_f, 'N_binsnq_e_E_f'),
                      (N_binsnq_p_rec_f, 'N_binsnq_p_rec_f'),(N_binsnq_p_RE_f, 'N_binsnq_p_RE_f'),(N_binsnq_p_R_f, 'N_binsnq_p_R_f'),(N_binsnq_p_E_f, 'N_binsnq_p_E_f'),
                      (N_binsnq_du_rec_f, 'N_binsnq_du_rec_f'),(N_binsnq_du_RE_f, 'N_binsnq_du_RE_f'),(N_binsnq_du_R_f, 'N_binsnq_du_R_f'),(N_binsnq_du_E_f, 'N_binsnq_du_E_f')]} 

        for carpeta, datos in datos_dict.items():
            for dato, nombre in datos:
                guardar_lista_en_carpeta(dato, nombre, carpeta)

#################################################################################################################################################
'Sacamos los valores calculados por grafmasterdata'
#################################################################################################################################################

if grafbins or grafdatacomp or grafchieffi or grafmarefi or grafcorrefici or grafchibi:
    "Obtengo la informacion calculada por masterchi"
    lista_N_eventos = listar_archivos_en_carpeta('N_eventos')
    lista_N_eventosnq = listar_archivos_en_carpeta('N_eventosnq')
    lista_N_eventos_f = listar_archivos_en_carpeta('N_eventos_f')
    lista_N_eventosnq_f = listar_archivos_en_carpeta('N_eventosnq_f')
    lista_t_data = listar_archivos_en_carpeta('t_data')
    
    N_data = leer_archivos_npy('N_eventos', lista_N_eventos)
    N_nq_data = leer_archivos_npy('N_eventosnq', lista_N_eventosnq)
    N_data_f = leer_archivos_npy('N_eventos_f', lista_N_eventos_f)
    N_nq_data_f = leer_archivos_npy('N_eventosnq_f', lista_N_eventosnq_f)
    t_data = leer_archivos_npy('t_data', lista_t_data)
    
    N_bins_prompt = N_data['N_bins_p_RE.npy']
    N_bins_de_e = N_data['N_bins_de_RE.npy']
    N_bins_de_u = N_data['N_bins_du_RE.npy'] 
    N_bins_delay = N_bins_de_e + N_bins_de_u
    
    # Datos para la grafica de comparativa 
    N_bins_rec,N_bins_rec_f = N_data['N_bins_rec.npy'],N_data_f['N_bins_rec_f.npy']
    N_bins_R,N_bins_R_f = N_data['N_bins_R.npy'],N_data_f['N_bins_R_f.npy']
    N_bins_E,N_bins_E_f = N_data['N_bins_E.npy'],N_data_f['N_bins_E_f.npy']
    N_bins_RE,N_bins_RE_f = N_data['N_bins_RE.npy'],N_data_f['N_bins_RE_f.npy']
    
    'Valores con FF '
    N_bins = N_data['N_bins_rec.npy']
    N_bins_de_rec = N_data['N_bins_de_rec.npy']
    N_bins_du_rec = N_data['N_bins_du_rec.npy']
    N_bins_d_rec = N_bins_du_rec + N_bins_de_rec
    N_bins_p_rec = N_data['N_bins_p_rec.npy']
    
    N_bins_RE = N_data['N_bins_RE.npy']
    N_bins_de_RE = N_data['N_bins_de_RE.npy']
    N_bins_du_RE = N_data['N_bins_du_RE.npy']
    N_bins_d_RE = N_bins_du_RE + N_bins_de_RE
    N_bins_p_RE = N_data['N_bins_p_RE.npy']

    N_bins_E = N_data['N_bins_E.npy']
    N_bins_R = N_data['N_bins_R.npy']
    N_binsnq = np.array([N_nq_data['N_binsnq_u_rec.npy'],N_nq_data['N_binsnq_e_rec.npy']])
    N_binsnq_E = np.array([N_nq_data['N_binsnq_u_E.npy'],N_nq_data['N_binsnq_e_E.npy']])
    N_binsnq_R = np.array([N_nq_data['N_binsnq_u_R.npy'],N_nq_data['N_binsnq_e_R.npy']])
    N_binsnq_RE = np.array([N_nq_data['N_binsnq_u_RE.npy'],N_nq_data['N_binsnq_e_RE.npy']])
    
    N_binsnq_3 = np.array([N_nq_data['N_binsnq_p_rec.npy'],N_nq_data['N_binsnq_du_rec.npy'],N_nq_data['N_binsnq_e_rec.npy']])
    N_binsnq_E_3 = np.array([N_nq_data['N_binsnq_p_E.npy'],N_nq_data['N_binsnq_du_E.npy'],N_nq_data['N_binsnq_e_E.npy']])
    N_binsnq_R_3 = np.array([N_nq_data['N_binsnq_p_R.npy'],N_nq_data['N_binsnq_du_R.npy'],N_nq_data['N_binsnq_e_R.npy']])
    N_binsnq_RE_3 = np.array([N_nq_data['N_binsnq_p_RE.npy'],N_nq_data['N_binsnq_du_RE.npy'],N_nq_data['N_binsnq_e_RE.npy']])

    'Valores sin FF '
    N_bins_f = N_data_f['N_bins_rec_f.npy']
    N_bins_de_rec_f = N_data_f['N_bins_de_rec_f.npy']
    N_bins_du_rec_f = N_data_f['N_bins_du_rec_f.npy']
    N_bins_d_rec_f = N_bins_du_rec_f + N_bins_de_rec_f
    N_bins_p_rec_f = N_data_f['N_bins_p_rec_f.npy']
    
    N_bins_R_f = N_data_f['N_bins_R_f.npy']
    N_bins_de_R_f = N_data_f['N_bins_de_R_f.npy']
    N_bins_du_R_f = N_data_f['N_bins_du_R_f.npy']
    N_bins_d_R_f = N_bins_du_R_f + N_bins_de_R_f
    N_bins_p_R_f = N_data_f['N_bins_p_R_f.npy']

    N_bins_E_f = N_data_f['N_bins_E_f.npy']
    N_bins_RE_f = N_data_f['N_bins_RE_f.npy']
    N_binsnq_f = np.array([N_nq_data_f['N_binsnq_u_rec_f.npy'],N_nq_data_f['N_binsnq_e_rec_f.npy']])
    N_binsnq_E_f = np.array([N_nq_data_f['N_binsnq_u_E_f.npy'],N_nq_data_f['N_binsnq_e_E_f.npy']])
    N_binsnq_R_f = np.array([N_nq_data_f['N_binsnq_u_R_f.npy'],N_nq_data_f['N_binsnq_e_R_f.npy']])
    N_binsnq_RE_f = np.array([N_nq_data_f['N_binsnq_u_RE_f.npy'],N_nq_data_f['N_binsnq_e_RE_f.npy']])        
    
    N_binsnq_f3 = np.array([N_nq_data_f['N_binsnq_p_rec_f.npy'],N_nq_data_f['N_binsnq_du_rec_f.npy'],N_nq_data_f['N_binsnq_e_rec_f.npy']])
    N_binsnq_E_f3 = np.array([N_nq_data_f['N_binsnq_p_E_f.npy'],N_nq_data_f['N_binsnq_du_E_f.npy'],N_nq_data_f['N_binsnq_e_E_f.npy']])
    N_binsnq_R_f3 = np.array([N_nq_data_f['N_binsnq_p_R_f.npy'],N_nq_data_f['N_binsnq_du_R_f.npy'],N_nq_data_f['N_binsnq_e_R_f.npy']])
    N_binsnq_RE_f3 = np.array([N_nq_data_f['N_binsnq_p_RE_f.npy'],N_nq_data_f['N_binsnq_du_RE_f.npy'],N_nq_data_f['N_binsnq_e_RE_f.npy']])
    
    if grafbins or grafdatacomp or grafchibi:
        if cortarPE:
            'Quitamos los elementos que no nos interesan'
            N_bins_prompt = N_bins_prompt[bin_corte:]
            N_bins_de_e = N_bins_de_e[bin_corte:]
            N_bins_de_u = N_bins_de_u[bin_corte:]
            N_bins_delay = N_bins_delay[bin_corte:]
            
            N_bins_rec,N_bins_rec_f = N_bins_rec[bin_corte:],N_bins_rec_f[bin_corte:]
            N_bins_R,N_bins_R_f =N_bins_R[bin_corte:],N_bins_R_f[bin_corte:]
            N_bins_E,N_bins_E_f = N_bins_E[bin_corte:],N_bins_E_f[bin_corte:]
            N_bins_RE,N_bins_RE_f = N_bins_RE[bin_corte:],N_bins_RE_f[bin_corte:] 
            
        g_d = t_data["g_d.npy"]
        g_p = t_data["g_p.npy"]
        DP_dtM_norm = t_data["DP_dtM_norm.npy"]
        DP_dtE_norm = t_data["DP_dtE_norm.npy"]

        'Calculo las listas de intervalos temporales (Cuidado que coincidan)'
        PE_array_rec = np.linspace(0, max(PE_coef), t_split_rec)
        tee_array_rec = PE_array_rec /LY
        t_array_rec = inverse_QFT_spline(PE_array_rec)
            
        if cortarPE:
            num_bins = 60 - bin_corte 
            rangosT = cortePE(t_max)
            t_array_rec,t_bins_rec = [],[]
            num_split_bins_rec = int(t_split_rec/num_bins)
            
            for i in range(len(rangosT) - 1):
                lsi=np.linspace(rangosT[i],rangosT[i+1],num_split_bins_rec,endpoint=False)
                t_array_rec.extend(lsi)
                t_bins_rec.append(lsi)       
            t_bins_rec = np.array(t_bins_rec)
            t_array_rec = np.array(t_array_rec)
            tee_array_rec = QF(t_array_rec)
            PE_array_rec = tee_array_rec *LY 
        else: 
            t_bins_rec = np.array_split(t_array_rec, num_bins) 
    
        #PE_bins = np.array_split(PE_array_rec, num_bins) 
        PE_bins = np.array_split(np.linspace(bin_corte,60,49*56), 56) 
        num_bins_temp = int(2*(timef-time0))        
        time_arr = np.linspace(time0,timef,timepass)
        time_bins = np.array_split(time_arr, num_bins_temp)

#################################################################################################################################################
'Realizamos la grafica del numero de bins'
#################################################################################################################################################
if grafbins:
    'Calculamos las funciones g_p y g_d pero ajustadas a esta figura'
    time_arrchi = np.linspace(time0,timef,len(DP_dtM_norm))
    amp_temp = np.array([1/8,1/8,1/8,1/8,1/8,1/8,1/8,1/8,1,2,2])/6*len(time_arrchi) 
    poscor = []

    for i in range(len(amp_temp)):
        if i == 0 : 
            poscor.append(int(amp_temp[i]))
        else:  
            poscor.append(int(amp_temp[i])+poscor[-1])
            
    time_binschi = np.array_split(time_arrchi, poscor)[:-1]
    DP_dtM_binschi = np.array_split(DP_dtM_norm, poscor)[:-1]
    DP_dtE_binschi = np.array_split(DP_dtE_norm, poscor)[:-1]
    
    g_pgraf = split_bins(len(DP_dtM_binschi), DP_dtM_binschi, time_binschi)
    g_dgraf = split_bins(len(DP_dtM_binschi), DP_dtE_binschi, time_binschi)

    'Calculo la info de los array bidimensionales'
    # Calculamos N_ij
    N_bins_2d_p = np.tensordot(g_pgraf,N_bins_prompt,axes=0) 
    N_bins_2d_du = np.tensordot(g_dgraf,N_bins_de_u,axes=0)
    N_bins_2d_de = np.tensordot(g_dgraf,N_bins_de_e,axes=0)
    N_bins_2d =  np.tensordot(g_pgraf,N_bins_prompt,axes=0) + np.tensordot(g_dgraf,N_bins_delay,axes=0) 

    #Calculamos los valores de las columnas
    N_bins_t_p = np.sum(N_bins_2d_p,axis=1)
    N_bins_T_p = np.sum(N_bins_2d_p,axis=0)
    N_bins_t_de = np.sum(N_bins_2d_de,axis=1)
    N_bins_T_de = np.sum(N_bins_2d_de,axis=0)
    N_bins_t_du = np.sum(N_bins_2d_du,axis=1)
    N_bins_T_du = np.sum(N_bins_2d_du,axis=0)
    N_bins_t = np.sum(N_bins_2d,axis=1)
    N_bins_T = np.sum(N_bins_2d,axis=0)

    'Agrupamos los bins para el histograma'
    N_bins_T_grop = []
    N_bins_T_grop_p = []
    N_bins_T_grop_du = []
    for i in range(int(5)):  # Añadirmos los intevalos de 4 bins
        N_bins_T_grop.append(np.sum(N_bins_T[i*4:i*4+4])/4)
        N_bins_T_grop_p.append(np.sum(N_bins_T_p[i*4:i*4+4])/4)
        N_bins_T_grop_du.append(np.sum(N_bins_T_du[i*4:i*4+4])/4)

    for j in range(0,3):     # Añadimos los intervalos de 8 bins 
        N_bins_T_grop.append(np.sum(N_bins_T[20+j*8:28+j*8])/8)
        N_bins_T_grop_p.append(np.sum(N_bins_T_p[20+j*8:28+j*8])/8)
        N_bins_T_grop_du.append(np.sum(N_bins_T_du[20+j*8:28+j*8])/8)
    
    # Añadimos el ultimo intervalo de 12 bins 
    N_bins_T_grop.append(np.sum(N_bins_T[44:])/12)
    N_bins_T_grop_p.append(np.sum(N_bins_T_p[44:])/12)
    N_bins_T_grop_du.append(np.sum(N_bins_T_du[44:])/12)    
    N_bins_T_grop_p = np.array(N_bins_T_grop_p)
    N_bins_T_grop_du = np.array(N_bins_T_grop_du)
        
    N_bins_t_0 = np.insert(N_bins_t, 0, 0)
    N_bins_t_p_0 = np.insert(N_bins_t_p, 0, 0)
    N_bins_t_de_0 = np.insert(N_bins_t_de, 0, 0)
    N_bins_t_du_0 = np.insert(N_bins_t_du, 0, 0)
    
    'Obtenemos los valores de la colaboración'
    conver_T = 10/515
    amp_T = np.array([4,4,4,4,4,8,8,8,12])
    p_T = np.array([17.,232.,254.,136.,45.,0.,0.,0.,0.])*conver_T
    de_T = np.array([0.,302.,466.,441.,345.,221.,108.,47.,22.])*conver_T
    du_T = np.array([0.,221.,304.,266.,194.,109.,42.,16.,0.])*conver_T
    Ntot_T = p_T + de_T + du_T
    
    conver_t = 100/493  
    amp_t = np.array([1/8,1/8,1/8,1/8,1/8,1/8,1/8,1/8,1,2,2])
    amp_t_bar = np.array([1/8,1/8,1/8,1/8,1/8,1/8,1/8,1/8,1/8,1,2,2])
    p_t = np.array([0,16,160,461,637,501,197,42,0,0,0])*conver_t
    de_t = np.array([0.,0.,18,87,212,324,376,378,290,140,52])*conver_t
    du_t = np.array([0.,0,0,48,114,190,216,217,168,80,30])*conver_t
    Ntot_t = (p_t + de_t + du_t)
    Ntot_t = np.insert(Ntot_t, 7, 141)
    
    'Calculamos las anchuras de los bins'
    cont = 4 
    mid_T,int_T = [],[]
    for i in range(len(amp_T)):
        mid_T.append((amp_T[i])/2 + cont)
        int_T.append((amp_T[i]) + cont)
        cont += amp_T[i]
        
    cont = 0   
    mid_t,int_t= [-0.625],[]
    for i in range(len(amp_t)):
        mid_t.append((amp_t[i])/2 + cont)
        int_t.append((amp_t[i]) + cont)
        cont += amp_t[i]
     
    'Añadimos un valor delante y detras para que quede bien graficado'
    mid_T_0 = np.array([-1]+ mid_T + [66]) 
    amp_t0 =np.insert(amp_t, 0, 1/8)
    Ntot_T = np.insert(Ntot_T, 0, 0)
    Ntot_T = np.append(Ntot_T, 0)
    mid_t_0 = np.array( mid_t + [9.])   
    mid_t_0 = np.insert(mid_t_0, 8, mid_t_0[8]+1/8)
    Ntot = np.insert(Ntot_t, 0, 0)
    Ntot_0 = np.insert(Ntot_t, 0, 0)
    Ntot_0 = np.append(Ntot_0, 0)
    N_bins_t = np.insert(N_bins_t, 0, 0)
    N_bins_t_du = np.insert(N_bins_t_du, 0, 0)
    N_bins_t_p = np.insert(N_bins_t_p, 0, 0)
    
#################################################################################################################################################
'Calculamos con los nuevos datos la distribucion chi^2 teniendo en cuenta la dis. temporal '
#################################################################################################################################################
if grafchieffi or grafmarefi or grafcorrefici:
    extremos_mu_rec = np.array([Q_2_SM*(1-ffu),Q_2_SM*(1+ffu)])     
    extremos_ee_rec  = np.array([Q_2_SM*(1-ffe),Q_2_SM*(1+ffe)])     
    q_array_mu_rec  = np.linspace(extremos_mu_rec[0], extremos_mu_rec[1],splitred) 
    q_array_ee_rec  = np.linspace(extremos_ee_rec[0], extremos_ee_rec[1],splitred)
    q_red_rec  = np.vstack((q_array_mu_rec , q_array_ee_rec ))
    q_meshgrid_rec  = np.meshgrid(*q_red_rec )
    qu_rec ,qe_rec  = q_meshgrid_rec   

    'Para el caso con FF y gausiano'
    chi_rec = chi_2q(q_meshgrid_rec,N_bins,N_binsnq,N_bins**0.5,False)
    chi_rec_E = chi_2q(q_meshgrid_rec,N_bins_E,N_binsnq_E,np.maximum(N_bins_E**0.5, 1),False)
    chi_rec_R = chi_2q(q_meshgrid_rec,N_bins_R,N_binsnq_R,N_bins_R**0.5,False)
    chi_rec_RE = chi_2q(q_meshgrid_rec,N_bins_RE,N_binsnq_RE,np.maximum(N_bins_RE**0.5, 1),False) 
       
    Z_rec= (chi_rec - np.min(chi_rec))
    Z_E= (chi_rec_E - np.min(chi_rec_E))
    Z_R= (chi_rec_R - np.min(chi_rec_R))
    Z_RE= (chi_rec_RE - np.min(chi_rec_RE))
    
    'Para el caso sin FF y gausiano'
    chi_rec_f = chi_2q(q_meshgrid_rec,N_bins_f,N_binsnq_f,N_bins_f**0.5,False)
    chi_rec_E_f = chi_2q(q_meshgrid_rec,N_bins_E_f,N_binsnq_E_f,np.maximum(N_bins_E_f**0.5, 1),False)
    chi_rec_R_f = chi_2q(q_meshgrid_rec,N_bins_R_f,N_binsnq_R_f,N_bins_R_f**0.5,False)
    chi_rec_RE_f = chi_2q(q_meshgrid_rec,N_bins_RE_f,N_binsnq_RE_f,np.maximum(N_bins_RE_f**0.5, 1),False)
    
    Z_rec_f= (chi_rec_f - np.min(chi_rec_f))
    Z_E_f= (chi_rec_E_f - np.min(chi_rec_E_f))
    Z_R_f= (chi_rec_R_f - np.min(chi_rec_R_f))
    Z_RE_f= (chi_rec_RE_f - np.min(chi_rec_RE_f))
    
    x_min_rec, y_min_rec = np.unravel_index(np.argmin(chi_rec),chi_rec.shape)
    x_min_E, y_min_E = np.unravel_index(np.argmin(chi_rec_E),chi_rec_E.shape)
    x_min_R, y_min_R = np.unravel_index(np.argmin(chi_rec_R),chi_rec_R.shape)
    x_min_RE, y_min_RE = np.unravel_index(np.argmin(chi_rec_RE),chi_rec_RE.shape)    

    if grafmarefi:
        'Marginalizo los valores'
        Z_RE_mu = np.min(Z_RE, axis=0)
        Z_RE_ee = np.min(Z_RE, axis=1)
        Z_mu = np.min(Z_rec_f, axis=0)
        Z_ee = np.min(Z_rec_f, axis=1)

        indice_RE_mu = np.array(np.where(Z_RE_mu < 1))
        indice_RE_ee = np.array(np.where(Z_RE_ee < 1))
        indice_mu = np.array(np.where(Z_mu < 1))
        indice_ee = np.array(np.where(Z_ee < 1))
        
    if grafcorrefici:
        elip_ind = np.where(Z_RE_f <= sigma)  
        elip_pnt = np.column_stack((qu_rec[elip_ind], qe_rec[elip_ind]))
        cov_efi,cov_diag_efi,trans_efi,ang_efi = cova(elip_pnt)
        std_devs_efi = np.sqrt(np.diag(cov_efi))
        std_matrix_efi = np.outer(std_devs_efi, std_devs_efi)
        corr_matrix_efi = cov_efi/std_matrix_efi
        
#################################################################################################################################################
'Calculamos la distribucion chi2 con bins temporales y espaciales'
#################################################################################################################################################
if grafchibi or grafmarbi or grafcorbi: 
    'Generamos el array de Q'
    extremos_mu_bi = np.array([Q_2_SM*(1-ffubi),Q_2_SM*(1+ffubi)])     
    extremos_ee_bi  = np.array([Q_2_SM*(1-ffebi),Q_2_SM*(1+ffebi)])     
    q_array_mu_bi  = np.linspace(extremos_mu_bi[0], extremos_mu_bi[1],splitred) 
    q_array_ee_bi  = np.linspace(extremos_ee_bi[0], extremos_ee_bi[1],splitred)
    q_red_bi  = np.vstack((q_array_mu_bi , q_array_ee_bi ))
    q_meshgrid_bi  = np.meshgrid(*q_red_bi )
    qu_bi ,qe_bi  = q_meshgrid_bi  
    
    'Calculamos los valores de N_ij'
    N_bins_2d_bi =  np.tensordot(g_p,N_bins_p_RE[bin_corte:],axes=0) + np.tensordot(g_d,N_bins_d_RE[bin_corte:],axes=0) 
    N_bins_2d_bi_rec =  np.tensordot(g_p,N_bins_p_rec[bin_corte:],axes=0) + np.tensordot(g_d,N_bins_d_rec[bin_corte:],axes=0) 
    N_bins_2d_bi_R_f =  np.tensordot(g_p,N_bins_p_R_f[bin_corte:],axes=0) + np.tensordot(g_d,N_bins_d_R_f[bin_corte:],axes=0) 
    N_bins_2d_bi_rec_f =  np.tensordot(g_p,N_bins_p_rec_f[bin_corte:],axes=0) + np.tensordot(g_d,N_bins_d_rec_f[bin_corte:],axes=0) 

    'Calculamos los coeficientes sin la Q'
    N_bins_2d_bi_nq = temporizador_bins(N_binsnq_RE_3,g_p,g_d)
    N_bins_2d_bi_rec_nq = temporizador_bins(N_binsnq_3,g_p,g_d) 
    N_bins_2d_bi_R_f_nq = temporizador_bins(N_binsnq_R_f3,g_p,g_d)
    N_bins_2d_bi_rec_f_nq = temporizador_bins(N_binsnq_f3,g_p,g_d)

    chi_RE_bi = chi_2q_bi(q_meshgrid_bi,N_bins_2d_bi,N_bins_2d_bi_nq,N_bins_2d_bi**0.5)
    chi_rec_bi = chi_2q_bi(q_meshgrid_bi,N_bins_2d_bi_rec,N_bins_2d_bi_rec_nq,N_bins_2d_bi_rec**0.5)    
    chi_R_f_bi = chi_2q_bi(q_meshgrid_bi,N_bins_2d_bi_R_f,N_bins_2d_bi_R_f_nq,N_bins_2d_bi_R_f**0.5)
    chi_f_bi = chi_2q_bi(q_meshgrid_bi,N_bins_2d_bi_rec_f,N_bins_2d_bi_rec_f_nq,N_bins_2d_bi_rec_f**0.5)    
    
    Z_RE_bi = (chi_RE_bi - np.min(chi_RE_bi))
    Z_rec_bi = (chi_rec_bi - np.min(chi_rec_bi))
    Z_R_f_bi = (chi_R_f_bi - np.min(chi_R_f_bi))
    Z_rec_f_bi = (chi_f_bi - np.min(chi_f_bi))
    
    x_min_RE_bi, y_min_RE_bi = np.unravel_index(np.argmin(chi_RE_bi),chi_RE_bi.shape)  
    x_min_rec_bi, y_min_rec_bi = np.unravel_index(np.argmin(chi_rec_bi),chi_rec_bi.shape)
    x_min_R_bi, y_min_R_bi = np.unravel_index(np.argmin(chi_R_f_bi),chi_R_f_bi.shape)
    x_min_E_bi, y_min_E_bi = np.unravel_index(np.argmin(chi_f_bi),chi_f_bi.shape)
    
    if grafmarbi:
        'Marginalizo los valores '
        Z_RE_mu_bi = np.min(Z_RE_bi, axis=0)
        Z_RE_ee_bi = np.min(Z_RE_bi, axis=1)
        Z_mu_bi = np.min(Z_rec_f_bi, axis=0)
        Z_ee_bi = np.min(Z_rec_f_bi, axis=1)

        indice_RE_mu_bi = np.array(np.where(Z_RE_mu_bi < 1))
        indice_RE_ee_bi = np.array(np.where(Z_RE_ee_bi < 1))
        indice_mu_bi = np.array(np.where(Z_mu_bi < 1))
        indice_ee_bi = np.array(np.where(Z_ee_bi < 1))
        
    if grafcorbi:
        elip_ind_bi = np.where(Z_RE_bi <= sigma)  
        elip_pnt_bi  = np.column_stack((qu_bi[elip_ind_bi], qe_bi[elip_ind_bi]))
        cov_efi_bi ,cov_diag_efi_bi ,trans_efi_bi ,ang_efi_bi  = cova(elip_pnt_bi )
        std_devs_efi_bi  = np.sqrt(np.diag(cov_efi_bi))
        std_matrix_efi_bi  = np.outer(std_devs_efi_bi , std_devs_efi_bi)
        corr_matrix_efi_bi  = cov_efi_bi /std_matrix_efi_bi 
        
if grafchicobi:
    'Generamos el array de Q'
    extremos_mu_rec = np.array([Q_2_SM*(1-ffu),Q_2_SM*(1+ffu)])     
    extremos_ee_rec  = np.array([Q_2_SM*(1-ffe),Q_2_SM*(1+ffe)])     
    q_array_mu_rec  = np.linspace(extremos_mu_rec[0], extremos_mu_rec[1],splitred) 
    q_array_ee_rec  = np.linspace(extremos_ee_rec[0], extremos_ee_rec[1],splitred)
    q_red_rec  = np.vstack((q_array_mu_rec , q_array_ee_rec ))
    q_meshgrid_rec  = np.meshgrid(*q_red_rec )
    qu_rec ,qe_rec  = q_meshgrid_rec  
    
    'Calculamos los valores de N_ij'
    N_bins_2d_bi =  np.tensordot(g_p,N_bins_p_RE[bin_corte:],axes=0) + np.tensordot(g_d,N_bins_d_RE[bin_corte:],axes=0) 
    N_bins_2d_bi_rec_f = np.tensordot(g_p,N_bins_p_rec_f[bin_corte:],axes=0) + np.tensordot(g_d,N_bins_d_rec_f[bin_corte:],axes=0) 
    
    'Calculamos los coeficientes sin la Q'
    N_bins_2d_bi_rec_f_nq = temporizador_bins(N_binsnq_f3,g_p,g_d)
    N_bins_2d_bi_nq = temporizador_bins(N_binsnq_RE_3,g_p,g_d)
    
    'Calculamos la funcion chi'
    chi_f_bi_com = chi_2q_bi(q_meshgrid_rec,N_bins_2d_bi_rec_f,N_bins_2d_bi_rec_f_nq,N_bins_2d_bi_rec_f**0.5)  
    chi_RE_bi_com = chi_2q_bi(q_meshgrid_rec,N_bins_2d_bi,N_bins_2d_bi_nq,np.maximum(N_bins_2d_bi**0.5**0.5, 1)) 
    chi_rec_f_com = chi_2q(q_meshgrid_rec,N_bins_f,N_binsnq_f,N_bins_f**0.5,False)
    chi_rec_RE_com = chi_2q(q_meshgrid_rec,N_bins_RE,N_binsnq_RE,np.maximum(N_bins_RE**0.5, 1),False) 
    
    'Minimizamos la función chi'
    Z_rec_f_bi_com = (chi_f_bi_com - np.min(chi_f_bi_com))
    Z_RE_bi_com = (chi_RE_bi_com - np.min(chi_RE_bi_com))
    Z_RE_com = (chi_rec_RE_com - np.min(chi_rec_RE_com))
    Z_rec_f_com = (chi_rec_f_com - np.min(chi_rec_f_com))        

#################################################################################################################################################
#################################################################################################################################################
'Realizamos las gráficas'
#################################################################################################################################################
#################################################################################################################################################

'Grafica del flujo energético'
if grafflux and Graf : 
    #fig17 = plt.figure('Flux')
    fig17 = plt.figure(num=None, figsize=(8, 4), dpi=190, facecolor='w', edgecolor='k')
    fig17.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
    fig17.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))  
    ax17 = plt.gca() 
    line171, = ax17.plot(Ener,impulse_signal, c='red', linewidth=1.5,label = r'$\frac{d \phi_{\nu_{\mu}}}{d E_{\nu}}$')
    line172, = ax17.plot(Ener,flux_vu, c='green', linewidth=1.5,label = r'$\frac{d \phi_{\bar{\nu}_{\mu}}}{d E_{\nu}}$')
    line173, = ax17.plot(Ener,flux_ve, c='blue', linewidth=1.5,label = r'$\frac{d \phi_{\nu_{e}}}{d E_{\nu}}$')
    ax17.set_title('Flujo de energia')
    ax17.set_ylabel(r'$\frac{d \phi_{\nu_{\ell}}}{d E_{\nu}}$ [MeV]', fontsize=18)
    ax17.set_xlabel(r' $ E_{\nu} \thinspace [MeV]$', fontsize=18)
    ax17.set_xlim(0.,Ener[-1])
    ax17.set_ylim(0,1e-8)
    ax17.grid(True, linestyle='--', alpha=0.8)
    #ax17.axis([0.,time_arr[-1],0.,1.1*np.max(DP_dtM_norm)])
    ax17.legend(fontsize=16, handlelength=1.5, handletextpad=1.5)
    if save == True: 
        plt.savefig('Flujo energetico.png')
    else:
        plt.show() 
        
'Grafica del flujo temporal'        
if graf_time and Graf :
    #fig13 = plt.figure('G_T')
    fig13 = plt.figure(num=None, figsize=(8, 4), dpi=190, facecolor='w', edgecolor='k')
    fig13.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
    fig13.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
    ax13 = plt.gca()  
    ax13.set_title(r'$\frac{dg}{dt}(t)$', fontsize=20)
    ax13.set_ylabel(r'$\frac{dP(t)}{dt}$  [A.u.]', fontsize=18)
    ax13.set_xlabel(r'Tiempo $[{\mu}s]$', fontsize=18)
    line131, = ax13.plot(time_arr, DP_dtM_norm, c='green', linewidth=1.5,label = r'$\frac{d P_{\nu_{\mu}}(t)}{dt}$' )
    line132, = ax13.plot(time_arr, DP_dtE_norm, c='red', linewidth=1.5,label = r'$\frac{d P_{\nu_{e},\nu_{\mu}}(t)}{dt}$')
    ax13.grid(True, linestyle='--', alpha=0.8)
    ax13.axis([0.,time_arr[-1],0.,1.1*np.max(DP_dtM_norm)])
    ax13.legend(fontsize=14, handlelength=1.5, handletextpad=1.5)
    if save == True: 
        plt.savefig('Flujo temporal.png')
    else:
        plt.show() 

'Grafica de DN/DT '
if graf2q == True:
    #fig = plt.figure('dN/dT 2q')
    fig = plt.figure(num=None, dpi=190, facecolor='w', edgecolor='k') # Para aumentar la grafica 
    fig.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
    fig.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
    ax0 = plt.gca()  
    ax0.set_title(r'$\frac{dN_{\alpha}}{dT_{nr}}(T_{nr})$', fontsize=18)
    ax0.set_ylabel(r'$\frac{dN_{\alpha}}{dT_{nr}}$ [$MeV^{-1}$]', fontsize=16)
    ax0.set_xlabel(r'$T_{nr} \thinspace [MeV]$', fontsize=16)
    line, = ax0.plot(t_array, dn_promt_list, c='green', linewidth=1.5 )
    line2, = ax0.plot(t_array, dn_delay_e_list, c='red', linewidth=1.5)
    line3, = ax0.plot(t_array, dn_delay_u_list, c='yellow', linewidth=1.5)
    line4, = ax0.plot(t_array, dn_delay_u_list + dn_delay_e_list, c='blue', linewidth=1.5)
    line5, = ax0.plot(t_array, dn_list, c='orange', linewidth=1.5)
    line6, = ax0.plot(t_array, dn_promt_listq2, linewidth=1.5, c='green', linestyle='--')
    line7, = ax0.plot(t_array, dn_delay_e_listq2, linewidth=1.5, c='red', linestyle='--')
    line8, = ax0.plot(t_array, dn_delay_u_listq2, linewidth=1.5, c='yellow', linestyle='--')
    line9, = ax0.plot(t_array, dn_delay_u_listq2 + dn_delay_e_listq2, c='blue', linestyle='--', linewidth=2.)
    line10, = ax0.plot(t_array, dn_listq2, c='orange', linewidth=1.5, linestyle='--')
    ax0.grid(True, linestyle='--', alpha=0.8)
    ax0.axis([0,tmax_delay, 0,np.max(dn_list)])
    legend_labels = [Line2D([0], [0], color='black', linestyle='-', label= r'$Q^{2}_{\mu}  = Q^{2}_{e}$'),
                     Line2D([0], [0], color='black', linestyle='--', label= r'$Q^{2}_{\mu} \neq Q^{2}_{e}$')]
    legend_labels.extend([
        Line2D([0], [0], marker='s', color='green', linestyle='None', label='Prompt ($\\nu_\\mu$)', markersize=8, markerfacecolor='green', markeredgewidth=0),
        Line2D([0], [0], marker='s', color='red', linestyle='None', label='Delayed ($\\nu_e$)', markersize=8, markerfacecolor='red', markeredgewidth=0),
        Line2D([0], [0], marker='s', color='yellow', linestyle='None', label='Delayed ($\\bar{\\nu}_\\mu$)', markersize=8, markerfacecolor='yellow', markeredgewidth=0),
        Line2D([0], [0], marker='s', color='blue', linestyle='None', label='Delayed (Total)', markersize=8, markerfacecolor='blue', markeredgewidth=0),
        Line2D([0], [0], marker='s', color='orange', linestyle='None', label='Total', markersize=8, markerfacecolor='orange', markeredgewidth=0),
    ])
    ax0.legend(handles=legend_labels, fontsize=16, handlelength=1.5, handletextpad=1.5)
    if save == True: 
        plt.savefig('DN_DT para T.png')
    else:
        plt.show() 
    
'Grafica que relaciona PE y T'
if grafTee and Graf : 
    fig5 = plt.figure('QF')
    #fig5.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.2e"))
    fig5.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.2e"))
    ax5 = plt.gca()  
    ax5.set_title(r'Relación entre PE y T [MeV] ', fontsize=20)
    ax5.set_xlabel(r'PE', fontsize=18)
    ax5.set_ylabel(r'T [MeV]', fontsize=18)
    line51, = ax5.plot(PE_array, t_array_qf, c='red', linewidth=1.5, label = 'Quenchin Factor')
    ax5.grid(True, linestyle='--', alpha=0.8)
    ax5.axis([0,1.01*np.max(PE_array),0,1.01*np.max(t_array_qf)])
    ax5.legend()
    if save == True: 
        plt.savefig('Relacion T y PE.png')
    else:
        plt.show()

'Grafica difenencial numero de eventos para PE en lugar de t'
if graf2q_Tee and Graf: 
    #fig9 = plt.figure('dN/dT 2q')
    fig9 = plt.figure(num=None, dpi=190, facecolor='w', edgecolor='k') # Para aumentar la grafica 
    #fig9.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
    #fig9.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
    ax9 = plt.gca() 
    ax9.set_title(r'$\frac{dN_{\alpha}}{dPE}(PE)$', fontsize=18)
    ax9.set_ylabel(r'$\frac{dN_{\alpha}}{dPE}$', fontsize=16)
    ax9.set_xlabel('PE', fontsize=16)
    line92, = ax9.plot(PE_array, dn_delay_e_list_pe, c='red', linewidth=1.5)
    line93, = ax9.plot(PE_array, dn_delay_u_list_pe, c='yellow', linewidth=1.5)
    line94, = ax9.plot(PE_array, dn_delay_u_list_pe + dn_delay_e_list_pe, c='blue', linewidth=1.5)
    line910, = ax9.plot(PE_array, dn_list2q_pe, c='orange', linewidth=1.5, linestyle='--')
    line95, = ax9.plot(PE_array, dn_list_pe, c='orange', linewidth=1.5)
    line97, = ax9.plot(PE_array, dn_delay_e_list2q_pe, linewidth=1.5, c='red', linestyle='--')
    line98, = ax9.plot(PE_array, dn_delay_u_list2q_pe, linewidth=1.5, c='yellow', linestyle='--')
    line96, = ax9.plot(PE_array, dn_promt_list2q_pe, linewidth=1.5, c='green', linestyle='--')
    line99, = ax9.plot(PE_array, dn_delay_u_list2q_pe + dn_delay_e_list2q_pe, c='blue', linestyle='--', linewidth=2.)
    line9, = ax9.plot(PE_array, dn_promt_list_pe, c='green', linewidth=1.5 )
    ax9.grid(True, linestyle='--', alpha=0.8)
    ax9.set_xlim(int(np.min(PE_array)),1.*np.max(PE_array))
    ax9.set_ylim(bottom=0)
    legend_labels = [Line2D([0], [0], color='black', linestyle='-', label= r'$Q^2_{\mu}  = Q^2_{e}$'),
                     Line2D([0], [0], color='black', linestyle='--', label= r'$Q^2_{\mu} \neq Q^2_{e}$')]
    legend_labels.extend([
        Line2D([0], [0], marker='s', color='green', linestyle='None', label='Prompt ($\\nu_\\mu$)', markersize=8, markerfacecolor='green', markeredgewidth=0),
        Line2D([0], [0], marker='s', color='red', linestyle='None', label='Delayed ($\\nu_e$)', markersize=8, markerfacecolor='red', markeredgewidth=0),
        Line2D([0], [0], marker='s', color='yellow', linestyle='None', label='Delayed ($\\bar{\\nu}_\\mu$)', markersize=8, markerfacecolor='yellow', markeredgewidth=0),
        Line2D([0], [0], marker='s', color='blue', linestyle='None', label='Delayed (Total)', markersize=8, markerfacecolor='blue', markeredgewidth=0),
        Line2D([0], [0], marker='s', color='orange', linestyle='None', label='Total', markersize=8, markerfacecolor='orange', markeredgewidth=0),
    ])
    ax9.legend(handles=legend_labels, fontsize=16, handlelength=1.5, handletextpad=1.5)
    if save == True: 
        plt.savefig('DN_DT para PE.png')
    else:
        plt.show()    
        
'Grafica de chi^2 unidimensional'
if grafchi1 and Graf: 
    #fig8 = plt.figure('chi2')
    fig8 = plt.figure(num=None, dpi=190, facecolor='w', edgecolor='k') 
    fig8.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
    fig8.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
    ax8 = plt.gca()  
    ax8.set_title(r'$\chi^{2} \left( Q^{2}_{W} \right) $ con bins energéticos', fontsize=18)
    ax8.set_ylabel(r'$\chi^{2}$', fontsize=16)
    ax8.set_xlabel(r'$Q^{2}_{W}$', fontsize=16)
    line82, = ax8.plot(q_array,chi_q_sqrI, c='red', linewidth=1.5, label=r'$\chi^{2}$ con dis. gaussiana')
    line83, = ax8.plot(q_array,chi_q_pos, c='blue', linewidth=1.5, label=r'$\chi^{2}$ con dis.  poissoniana')
    line84 = ax8.axvline(x=Q_2_SM, color='black', linestyle='--',label='Valor SM')
    ax8.axvspan(Q_2_sigma_sqr[0], Q_2_sigma_sqr[-1], color='red', alpha=0.1, linewidth=2, label=r'1 $\sigma$ con dis. gaussiana')
    ax8.axvspan(Q_2_sigma_pos[0], Q_2_sigma_pos[-1], color='blue', alpha=0.1, linewidth=2, label=r'1 $\sigma$ con dis. poissoniana')
    ax8.grid(True, linestyle='--', alpha=0.8)
    ax8.axis([q_array[0],q_array[-1],0,1.01*np.max(chi_q_sqrI)])
    ax8.legend()
    plt.plot()

'Grafica del chi^2 con la region de 1 sigma'
if grafchi2q  and Graf :
    title2q = r'$\chi^{2} \left( \tilde{Q}_{\mu}^{2}, \tilde{Q}_{e}^{2} \right) $ con bins energéticos'
    label_chi_sqr = r'$\chi²$ gaussiano'
    label_chi_pos = r'$\chi²$ poissoniana'
    
    fig4 = plt.figure(num=None, dpi=190, facecolor='w', edgecolor='k') 
    ax4 = plt.gca()
    contour_sqr = ax4.contourf(qu, qe, Z_sqr, levels=[0, sigma], colors=['blue'], alpha=0.25)
    contour_pos = ax4.contourf(qu, qe, Z_pos, levels=[0, sigma], colors=['red'], alpha=0.25)
    scatter_sm = ax4.scatter(Q_2_SM, Q_2_SM, color='purple', marker='*', s=25, label='Valor SM')
    scatter_sqr = ax4.scatter(qu[x_min_sqr,y_min_sqr], qe[x_min_sqr,y_min_sqr], color='blue', marker='*', s=25, label='Mínimo gaussiano')
    scatter_pos = ax4.scatter(qu[x_min_pos,y_min_pos], qe[x_min_pos,y_min_pos], color='red', marker='*', s=25, label='Mínimo poissoniana')
    
    ax4.set_xlabel(r'$\tilde{Q}_{\mu}^{2}$', fontsize=16)
    ax4.set_ylabel(r'$\tilde{Q}_{e}^{2}$', fontsize=16)
    ax4.set_title(title2q, fontsize=18)        
    ax4.axis([extremos_mu[0],extremos_mu[1],extremos_ee[0],extremos_ee[1]])
    ax4.grid(True, linestyle='--', alpha=0.8)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='blue', markeredgecolor='none', markersize=10),
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='red', markeredgecolor='none', markersize=10),
        plt.Line2D([0], [0], marker='*', color='none', markerfacecolor='purple', markeredgecolor='none', markersize=10),
        plt.Line2D([0], [0], marker='*', color='none', markerfacecolor='blue', markeredgecolor='none', markersize=10),
        plt.Line2D([0], [0], marker='*', color='none',  markerfacecolor='red', markeredgecolor='none', markersize=10)]
    ax4.legend(legend_elements, [label_chi_sqr, label_chi_pos, 'Valor SM', 'Mínimo gaussiano', 'Mínimo poissoniana'],fontsize=12)
    if save == True: 
        plt.savefig('chi2 bidimensional.png')
    else:
        plt.show() 

if grafmini1d and Graf :
    fig2, (ax21, ax22) = plt.subplots(2, 1, dpi=170)
    fig2.suptitle(r'Marginalización de $\chi^{2} (\tilde{Q}_{\mu}^{2},\tilde{Q}_{e}^{2})$  con bins energéticos', fontsize=18)
    ax21.set_ylabel(r'$\chi^{2} (\tilde{Q}_{\mu}^{2})$', fontsize=16)  
    ax21.set_xlabel(r'$\tilde{Q}_{\mu}^{2}$', fontsize=16)  
    line22, = ax21.plot(q_array_mu, chix_sqr, c='red', linewidth=1.5, label=r'$\chi^{2}$ con dis. gaussiana')
    line23, = ax21.plot(q_array_mu, chix_pos, c='green', linewidth=1.5, label=r'$\chi^{2}$ con dis. poisson')
    line84 = ax21.axvline(x=Q_2_SM, color='black', linestyle='--',label='Valor SM')
    ax21.axvspan(Q_2_sigma_sqr_2qu[0], Q_2_sigma_sqr_2qu[-1], color='red', alpha=0.15, linewidth=2, label=r'1 $\sigma$ con dis. gaussiana')
    ax21.axvspan(Q_2_sigma_pos_2qu[0], Q_2_sigma_pos_2qu[-1], color='blue', alpha=0.15, linewidth=2, label=r'1 $\sigma$ con dis. poisson')
    ax21.axis([3400.,7900.,-np.max(chix_pos)*0.00,5.])
    ax21.grid(True, linestyle='--', alpha=0.8)
    ax21.legend(loc='upper center', fontsize=10)
    
    ax22.set_ylabel(r'$\chi^{2} (\tilde{Q}_{e}^{2})$', fontsize=16)  
    ax22.set_xlabel(r'$\tilde{Q}_{e}^{2}$', fontsize=16)   
    line22, = ax22.plot(q_array_ee, chiy_sqr, c='red', linewidth=1.5, label=r'$\chi^{2}$ con dis. gaussiana')
    line23, = ax22.plot(q_array_ee, chiy_pos, c='green', linewidth=1.5, label=r'$\chi^{2}$ con dis. poisson')
    line84 = ax22.axvline(x=Q_2_SM, color='black', linestyle='--',label='Valor SM')
    ax22.axvspan(Q_2_sigma_sqr_2qe[0], Q_2_sigma_sqr_2qe[-1], color='red', alpha=0.15, linewidth=2, label=r' 1 $\sigma$ con dis. gaussiana')
    ax22.axvspan(Q_2_sigma_pos_2qe[0], Q_2_sigma_pos_2qe[-1], color='blue', alpha=0.15, linewidth=2, label=r'1 $\sigma$ con dis. poisson')
    ax22.axis([extremos_ee[0],extremos_ee[1],0,5.])
    ax22.grid(True, linestyle='--', alpha=0.8)
    ax22.legend(loc='upper center', fontsize=10)
    if save == True: 
        plt.savefig('chi2 marginalizado.png')
    else:
        plt.show() 

'Grafica en 3 dimensiones del chi^2 de dos variables'
if grafchi2q_3d and Graf:
    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    nombres = [r'$\chi^{2}$ (error de $\sqrt{N}$)', r'$\chi^{2}$ (dis. poisson)']
    cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.05])  
    normas = [chi_q_sqr_2q-np.min(chi_q_sqr_2q),chi_q_pos_2q-np.min(chi_q_pos_2q)]
    for ax, normchi_q, i in zip(axs, normas,range(len(nombres)) ):
        surface = ax.plot_surface(qu, qe, normchi_q, cmap='plasma')
        ax.set_xlabel(r'$\tilde{Q}_{\mu}$')
        ax.set_ylabel(r'$\tilde{Q}_{e}$')
        ax.set_zlabel(r'$\chi^{2}$')
        ax.set_title(nombres[i])
    fig.colorbar(surface, cax=cbar_ax, orientation='horizontal', label=r'$\chi^{2}$')
    if save == True: 
        plt.savefig('chi2 en 3 dimensiones.png')
    else:
        plt.show() 

'Grafica de la elipse tras hacer la combianción lineal'
if graffcorr and Graf : 
    title = r'$\chi^{2} (\tilde{Q}_{e}^{2},\tilde{Q}_{\mu}^{2}) $ ' + 'con {0} bins'.format(num_bins)
    y_label = r'$\tilde{Q}_{+}^{2}$ = ' + '{}'.format(round(a_coef,2)) + r'$\tilde{Q}_{e}^{2}$ + ' + '{}'.format(round(b_coef,2)) + r'$\tilde{Q}_{\mu}^{2}$' 
    x_label = r'$\tilde{Q}_{-}^{2}$ = ' + '{}'.format(round(b_coef,2)) + r'$\tilde{Q}_{e}^{2}$ - ' + '{}'.format(round(a_coef,2))  + r'$\tilde{Q}_{\mu}^{2}$'
    
    fig10, ax10 = plt.subplots(1, 2,dpi = 190)
    line101, = ax10[0].plot(q_array_mu,indeterminado, c='yellow', linewidth=1.5, label=r'Eje indeterminado')
    line102, = ax10[0].plot(q_array_mu,determinado, c='red', linewidth=1.5, label=r'Eje determinado')
    contour_sqr = ax10[0].contourf(qu, qe, Z_sqr, levels=[0, sigma], colors=['blue'], alpha=0.25)
    ax10[0].set_xlabel(r'$\tilde{Q}_{\mu}^{2}$', fontsize=14)
    ax10[0].set_ylabel(r'$\tilde{Q}_{e}^{2}$', fontsize=14)
    ax10[0].set_title(r'$\chi^{2} (\tilde{Q}_{\mu}^{2},\tilde{Q}_{e}^{2})$', fontsize=16)
    ax10[0].axis([extremos_mu[0],extremos_mu[1],extremos_ee[0],extremos_ee[1]])
    ax10[0].grid(True, linestyle='--', alpha=0.8)
    scatter_sm = ax10[0].scatter(Q_2_SM, Q_2_SM, color='purple', marker='*', s=25, label='Valor SM')
    scatter_pos = ax10[0].scatter(qu[x_min_pos, y_min_pos], qe[x_min_pos, y_min_pos], color='red', marker='*', s=25, label='Mínimo poissoniana')
    legend_elements1 = [
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='blue', markeredgecolor='none', markersize=10),
        plt.Line2D([0], [0], marker='*', color='none', markerfacecolor='purple', markeredgecolor='none', markersize=10),
        plt.Line2D([0], [0], marker='*', color='none', markerfacecolor='red', markeredgecolor='none', markersize=10),   
        line101,line102]
    ax10[0].legend(legend_elements1, [r'1 $\sigma$ de $\chi^2$ ', 'Valor SM', r'Mínimo $\chi^2$', 'Eje preciso', 'Eje impreciso'])
    scatter_sm = ax10[0].scatter(Q_2_SM, Q_2_SM, color='purple', marker='*', s=25, label='Valor SM')
    scatter_pos = ax10[0].scatter(qu[x_min_pos, y_min_pos], qe[x_min_pos, y_min_pos], color='red', marker='*', s=25, label='Mínimo poissoniana')
    
    line103, = ax10[1].plot(eje_indet[1],eje_indet[0], c='red', linewidth=1.5, label=r'Eje mayor')
    line104, = ax10[1].plot(eje_det[1],eje_det[0], c='yellow', linewidth=1.5, label=r'Eje menor')
    ax10[1].plot(elip_cor_sqr[1], elip_cor_sqr[0], color='blue', alpha=0.25)
    ax10[1].set_xlabel(x_label, fontsize=14)
    ax10[1].set_ylabel(y_label, fontsize=14)
    ax10[1].set_title(r'$\chi^{2} (\tilde{Q}_{+}^{2},\tilde{Q}_{-}^{2})$', fontsize=16)
    ax10[1].legend(fontsize=14)
    ax10[1].axis([eje_det[1][0],eje_det[1][-1],Q_2_SM_cx*(1-0.10),Q_2_SM_cx*(1+0.10)])
    ax10[1].grid(True, linestyle='--', alpha=0.8)
    
    legend_elements2 = [
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='blue', markeredgecolor='none', markersize=10),
        plt.Line2D([0], [0], marker='*', color='none', markerfacecolor='purple', markeredgecolor='none', markersize=10),
        plt.Line2D([0], [0], marker='*', color='none', markerfacecolor='red', markeredgecolor='none', markersize=10),     
        line103,line104]
    ax10[1].legend(legend_elements2, [r'1 $\sigma$ de $\chi^2$ ', 'Valor SM', r'Mínimo $\chi^2$', 'Eje preciso', 'Eje impreciso'])
    scatter_sm = ax10[0].scatter(Q_2_SM, Q_2_SM, color='purple', marker='*', s=25, label='Valor SM')
    scatter_pos = ax10[0].scatter(qu[x_min_pos, y_min_pos], qe[x_min_pos, y_min_pos], color='red', marker='*', s=25, label='Mínimo poissoniana')
    scatter_sm_c = ax10[1].scatter(Q_2_SM_cy, Q_2_SM_cx, color='purple', marker='*', s=25, label='Valor modelo estándar')
    if save == True: 
        plt.savefig('chi2 con comb lineal .png')
    else:
        plt.show() 
        
if graf_FF and Graf:
    #fig16 = plt.figure('FF')
    fig16 = plt.figure(num=None, figsize=(8, 4), dpi=190, facecolor='w', edgecolor='k')
    fig16.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
    fig16.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))  
    ax16 = plt.gca() 
    ax16.set_title(r'Factor de forma nuclear $\mathcal{F}$',fontsize=18)
    ax16.set_ylabel(r'$\mathcal{F}^{2}$', fontsize=16)
    ax16.set_xlabel(r'T [MeV]', fontsize=16)
    line161, = ax16.plot(t_array, F_p**2, c='green', linewidth=1.5,label = r'$\mathcal{F}_{p}^{2}$' )
    line162, = ax16.plot(t_array, F_n**2, c='red', linewidth=1.5,label = r'$\mathcal{F}_{n}^{2}$')
    line162, = ax16.plot(t_array, F_tot**2, c='purple', linewidth=1.5,label = r'$\mathcal{F}_{SM}^{2}$')
    ax16.grid(True, linestyle='--', alpha=0.8)
    ax16.axis([t_array[0],t_array[-1],0.,1.])
    ax16.legend(fontsize=16, handlelength=1.5, handletextpad=1.5)
    if save == True: 
        plt.savefig('Factor de forma.png')
    else:
        plt.show() 

'Grafica de compendio de funciones chi^2 como un mapa topologico'
if graftopo:
    ancarta = leer_datos_de_carpeta('datos_elipses')

    'Genero la paleta de colores automatica '
    color_dict = {}
    unique_bins = sorted(set(ancarta[3]))  # Obtener números de bins
    cmap = plt.colormaps['tab20']          # paleta de colores 
    for i, bins in enumerate(unique_bins):
        color_dict[bins] = cmap(i)
    
    'Genero las leyendas '
    legend_elements1 = [plt.Line2D([0], [0], marker='none', color='blue', markerfacecolor='black', linestyle='-', lw=2,label=r'$\chi^{2}$ con $\mathcal{F}^{2}(q^2)$'),
        plt.Line2D([0], [0], marker='none', color='blue', markerfacecolor='black', linestyle='--', lw=2,label=r'$\chi^{2}$ con $\mathcal{F}^{2}(q^2)$'),
        plt.Line2D([0], [0], marker='*', color='none', markerfacecolor='red', markeredgecolor='none', markersize=10,label=r'Minimo de $\chi^{2}$'),
        plt.Line2D([0], [0], marker='*', color='none', markerfacecolor='purple', markeredgecolor='none', markersize=10,label='Minimo S.M.'),]
    legend_elements2 = [plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=color_dict[bins], markersize=10, label=f'{bins} bins') for bins in unique_bins]
    legend_elements = legend_elements2 + legend_elements1
    legend_label1 = [r'$\chi^{2}$ sin $\mathcal{F}^{2}(q^2)$',r'$\chi^{2}$ con $\mathcal{F}^{2}(q^2)$', r'Minimo de $\chi^{2}$','Minimo S.M.']
    legend_label2 = [f'{bins} bins'for bins in unique_bins]
    legend_label = legend_label2 + legend_label1
    
    fig18, (ax181, ax182) = plt.subplots(1, 2, figsize=(12, 6))
    ax181.set_title('Mínimo gaussiano')
    for j in range(len(ancarta[0])):
        bins = ancarta[3][j]
        if ancarta[2][j]:
            linestyle = '--'  # Línea punteada para "con"
            labelss = f'{bins} bins con F'
        else:
            linestyle = '-'  # Línea sólida para "sin"
            labelss = f'{bins} bins sin F'
        color = color_dict[bins]
        contour_sqr = ax181.contour(qu, qe, ancarta[0][j], levels=[0, sigma], colors=[color]*2, alpha=1, linestyles=linestyle)
        
    a1 = np.array(ancarta[0][j])
    x_min_sqr, y_min_sqr = np.unravel_index(np.argmin(a1), a1.shape)
    ax181.scatter(qu[x_min_sqr, y_min_sqr], qe[x_min_sqr, y_min_sqr], color='red', marker='*', s=50, label='Mínimo gaussiano')
    ax181.scatter(Q_2_SM,Q_2_SM, color='purple', marker='*', s=50, label='Mínimo modelo estandar')
    ax181.set_xlabel(r'$\tilde{Q}_{\mu}^{2}$', fontsize=20)
    ax181.set_ylabel(r'$\tilde{Q}_{e}^{2}$', fontsize=20)
    ax181.axis([extremos_mu[0], extremos_mu[1], extremos_ee[0], extremos_ee[1]])
    ax181.grid(True, linestyle='--', alpha=0.8)
    ax181.legend(legend_elements,legend_label)

    ax182.set_title('Mínimo poissoniana')
    for j in range(len(ancarta[1])):
        bins = ancarta[3][j]
        if ancarta[2][j]:
            linestyle = '--'  # Línea punteada para "con"
            labelss = f'{bins} bins con F'
        else:
            linestyle = '-'  # Línea sólida para "sin"
            labelss = f'{bins} bins sin F'
        color = color_dict[bins]
        contour_pos = ax182.contour(qu, qe, ancarta[1][j], levels=[0, sigma], colors=[color]*2, alpha=1, linestyles=linestyle)
    b2 = np.array(ancarta[1][j])
    x_min_pos, y_min_pos = np.unravel_index(np.argmin(b2), b2.shape)
    ax182.scatter(qu[x_min_pos, y_min_pos], qe[x_min_pos, y_min_pos], color='red', marker='*', s=50, label='Mínimo poissoniana')
    ax182.scatter(Q_2_SM,Q_2_SM, color='purple', marker='*', s=50, label='Mínimo SM ')
    ax182.set_xlabel(r'$\tilde{Q}_{\mu}^{2}$', fontsize=20)
    ax182.set_ylabel(r'$\tilde{Q}_{e}^{2}$', fontsize=20)
    ax182.axis([extremos_mu[0], extremos_mu[1], extremos_ee[0], extremos_ee[1]])
    ax182.grid(True, linestyle='--', alpha=0.8)
    ax182.legend(legend_elements,legend_label)
    fig18.suptitle('Valores 1 sigma en función del Nº bins y FF', fontsize=22)
    if save == True: 
        plt.savefig('chi2 topografico .png')
    else:
        plt.show() 

'Grafica de resolucion energetica'
if grafresol and Graf  :
    fig21, (ax211, ax212) = plt.subplots(1, 2, figsize=(8, 3),dpi=190)
    R_matrix =  np.clip(R_matrix, None, 1)
    R_lim = [np.min(tee_array_rec)*1e3,np.max(tee_array_rec)*1e3,np.min(t_array_rec)*1e3, np.max(t_array_rec)*1e3]
    heatmap = ax211.imshow(R_matrix, extent= R_lim , aspect='auto', origin='lower', cmap='binary')
    cbar = fig21.colorbar(heatmap, ax=ax211)
    ax211.set_xlabel(r'$T_{ee}^{rec}$ [keV]', fontsize=14)
    ax211.set_ylabel(r'$T_{nr}$ [keV]', fontsize=14)
    ax211.set_title(r'$\mathcal{R}(T_{ee}^{rec},T_{ee}(T_{nr}))$', fontsize=18)
    ax211.set_ylim(t_array_rec[0]*1e3, t_array_rec[-1]*1e3)
    ax211.set_xlim(tee_array_rec[0]*1e3, tee_array_rec[-1]*1e3)
    ax211.grid(True, alpha=0.3)

    ax212.set_xlabel(r'$T_{nr}$ [keV]', fontsize=14)
    ax212.set_ylabel(r'$\mathcal{R}(T_{ee}^{rec},T_{ee}(T_{nr}))$' ,fontsize=14)
    ax212.set_title('$\mathcal{R}(T_{ee}^{rec},T_{ee}(T_{nr}))$',fontsize=18)
    line211, = ax212.plot(t_array_rec*1e3, R_15, c='Red', linewidth=1.5, label=r'$\mathcal{R}$ para PE $= 15$')
    line212, = ax212.plot(t_array_rec*1e3, R_20, c='blue', linewidth=1.5, label=r'$\mathcal{R}$ para PE $= 20$')
    line212, = ax212.plot(t_array_rec*1e3, R_30, c='green', linewidth=1.5, label=r'$\mathcal{R}$ para PE $= 30$')
    line213, = ax212.plot(t_array_rec*1e3, R_40, c='orange', linewidth=1.5, label=r'$\mathcal{R}$ para PE $= 40$')
    ax212.grid(True, alpha=0.3)
    ax212.set_xlim(t_array_rec[0]*1e3, t_array_rec[-1]*1e3)
    ax212.set_ylim(bottom = 0)
    ax212.legend(fontsize=14)
    
    plt.tight_layout()
    if save == True: 
        plt.savefig('resolucion energetica.png')
    else:
        plt.show() 

'Grafica de las trasnformaciones'
if grafJacob and Graf: 
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].plot(tee_array_rec_k, inverse_QF_k_spline(tee_array_rec_k))
    axs[0, 0].set_title("Inverse QF_k Spline (Tee Array Rec_k)")
    axs[0, 0].set_xlabel("Tee ")
    axs[0, 0].set_ylabel("T")
    
    axs[0, 1].plot(t_array_rec_k, QF_k(t_array_rec_k))
    axs[0, 1].set_title("QF_k (T Array Rec_k)")
    axs[0, 1].set_xlabel("T ")
    axs[0, 1].set_ylabel("Tee")
    
    axs[1, 0].plot(PE_array_rec, inverse_QFT_k_spline(PE_array_rec))
    axs[1, 0].set_title("Inverse QF_k Spline (PE Array Rec)")
    axs[1, 0].set_xlabel("PE ")
    axs[1, 0].set_ylabel("T")
    
    axs[1, 1].plot(t_array_rec_k, QFT_k(t_array_rec_k))
    axs[1, 1].set_title("QFT_k (T Array Rec_k)")
    axs[1, 1].set_xlabel("T ")
    axs[1, 1].set_ylabel("PE")
    if save == True: 
        plt.savefig('transformaciones.png')
    else:
        plt.show() 

'Grafica de los jacobianos'
if grafJacob and Graf: 
    fig2, axs2 = plt.subplots(2, 2, figsize=(12, 10))
    axs2[0, 0].plot(t_array_rec_k,J_T_TE)
    axs2[0, 0].set_title("Jacobiano de Tee a T")
    axs2[0, 0].set_xlabel("T ")
    axs2[0, 0].set_ylabel("Jacobiano de Tee a T")
    
    axs2[0, 1].plot(t_array_rec_k,J_T_PE)
    axs2[0, 1].set_title("Jacobiano de Pe a T")
    axs2[0, 1].set_xlabel("T ")
    axs2[0, 1].set_ylabel("Jacobiano de Pe a T")
    
    axs2[1, 0].plot(tee_array_rec_k,J_TE_T)
    axs2[1, 0].set_title("Jacobiano de T a te")
    axs2[1, 0].set_xlabel("te")
    axs2[1, 0].set_ylabel("Jacobiano de T a te")
    
    axs2[1, 1].plot(PE_array_rec,J_PE_T)
    axs2[1, 1].set_title("Jacobiano de T a Pe")
    axs2[1, 1].set_xlabel("PE")
    axs2[1, 1].set_ylabel("Jacobiano de T a Pe")
    if save == True: 
        plt.savefig('jacobianos.png')
    else:
        plt.show()

'Grafica de las eficiencias energeticas y temporales'
if grafefici and Graf :
    E_te = efpe(PE_array_rec) 
    E_t = eft(time_arrchi) 
    fig22, (ax221, ax222) = plt.subplots(1, 2, figsize=(9, 3),dpi=190)
    ax221.set_xlabel(r'PE', fontsize=16)
    ax221.set_ylabel(r'$\epsilon_E$', fontsize=16)
    ax221.set_title(r'Eficiencia energética $\epsilon_E(PE)$ ', fontsize=18)
    ax221.plot(PE_array_rec, E_te, c='black', linewidth=1.5, label=r'Eficiencia energía')
    ax221.set_xlim(np.min(PE_array_rec), np.max(PE_array_rec))
    ax221.set_ylim(0.,1.)
    ax221.grid(True, alpha=0.3)

    ax222.set_xlabel(r'Tiempo $[{\mu}s]$', fontsize=16)
    ax222.set_ylabel(r'$\epsilon_{t}$' ,fontsize=16)
    ax222.set_title('Eficiencia temporal $\epsilon_{t}(t)$',fontsize=18)
    ax222.plot(time_arrchi, E_t, c='black', linewidth=1.5, label=r'Eficiencia temporal')
    ax222.grid(True, alpha=0.3)
    ax222.axis([np.min(time_arrchi), np.max(time_arrchi), 0., 1.])
    ax222.set_ylim(0.,1.)
    
    plt.tight_layout()
    if save == True: 
        plt.savefig('eficiencias.png')
    else:
        plt.show()

'Realizamos un plot de los eventos obtenido en cada bien de forma similar a coherent'
if grafbins and Graf :
    fig14, (ax141, ax142) = plt.subplots(1, 2,figsize = (8,12), dpi = 190)
    ax141.bar(mid_t, N_bins_t/amp_t0, width=amp_t0, align='center', color='yellow', alpha=1, label=r'$\nu_{e}$')
    ax141.bar(mid_t, N_bins_t_du/amp_t0+N_bins_t_p/amp_t0, width=amp_t0, align='center', color='orange', alpha=1, label=r'$\bar{\nu}_{\mu}$')
    ax141.bar(mid_t, N_bins_t_p/amp_t0, width=amp_t0,align='center', color='brown', alpha=1, label=r'$\nu_{\mu}$')
    bar_mid_t = mid_t_0 - np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1, 0.25, -0.05, 0.45, 0.10, -0.15, 0.15])
    ax141.plot(bar_mid_t, Ntot_0, drawstyle='steps-mid', color='black', linewidth=2.0, label=r'COHERENT')
    ax141.set_xlabel(r'Tiempo [$\mu$s]',fontsize=16)
    ax141.set_ylabel(r'Nº eventos / $\mu$ s',fontsize=16)
    ax141.set_title('Distribución temporal',fontsize=18)
    ax141.set_xlim(-0.25,6)
    ax141.grid(True, alpha=0.3)
    ax141.legend(fontsize=14)
    
    ax142.bar(mid_T, N_bins_T_grop, width=amp_T, align='center', color='yellow', alpha=1, label=r'$\nu_{e}$')
    ax142.bar(mid_T, N_bins_T_grop_du + N_bins_T_grop_p, width=amp_T, align='center', color='orange', alpha=1, label=r'$\bar{\nu}_{\mu}$')
    ax142.bar(mid_T, N_bins_T_grop_p, width=amp_T,align='center', color='brown', alpha=1, label=r'$\nu_{\mu}$')
    bar_mid_T = mid_T_0 - np.array([0.3, 0.94, -0.4, 0.2, 0.2, -0.1, 2.0, -1.5, 1.4, 0.3, -0.9]) # Correccion ancho de barra
    ax142.plot(bar_mid_T, Ntot_T, drawstyle='steps-mid', color='black', linewidth=2.0, label=r'COHERENT')
    ax142.set_xlabel('PE' ,fontsize=16)
    ax142.set_ylabel(r'Nº eventos / PE' ,fontsize=16)
    ax142.set_title('Distribución energética',fontsize=18)
    ax142.grid(True, alpha=0.3)
    ax142.set_xlim(bin_corte, 60)
    ax142.legend(fontsize=14)
    if save == True: 
        plt.savefig('histograma eventos.png')
    else:
        plt.show()

'Comparacion del numero de eventos en funcion de si se aplica o no efectos experimentales'
if grafdatacomp:
    Pe_range = np.arange(0,60,1) 
    fig20 = plt.figure(num=None, dpi=190, facecolor='w', edgecolor='k')
    #fig20 = plt.figure(num=None, figsize=(8, 4), dpi=190, facecolor='w', edgecolor='k')
    ax20 = plt.gca() 
    ax20.set_title(r'Variación del nº eventos ', fontsize=18)
    ax20.set_ylabel(r'Número de eventos', fontsize=16)
    ax20.set_xlabel(r'Bin energético', fontsize=16)
    scatter1 = ax20.scatter(Pe_range, N_bins_rec, c='red', marker='^', s=15, label=r'Sin incluir $\mathcal{R}$ ni $\epsilon$')
    scatter2 = ax20.scatter(Pe_range, N_bins_R, c='blue', marker='^', s=15, label=r'Incluyendo $\mathcal{R}$ pero no $\epsilon$')
    scatter4 = ax20.scatter(Pe_range, N_bins_RE, c='green', marker='^', s=15, label=r'Incluyendo $\mathcal{R}$ y $\epsilon$')
    scatter5 = ax20.scatter(Pe_range, N_bins_rec_f, c='red', marker='s', s=15, label=r'Sin incluir $\mathcal{R}$ ni $\epsilon$')
    scatter6 = ax20.scatter(Pe_range, N_bins_R_f, c='blue', marker='s', s=15, label=r'Incluyendo $\mathcal{R}$ pero no $\epsilon$')
    scatter8 = ax20.scatter(Pe_range, N_bins_RE_f, c='green', marker='s', s=15, label=r'Incluyendo $\mathcal{R}$ y $\epsilon$')
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='red', markersize=15, linestyle='None', label=r'Sin incluir $\mathcal{R}$ ni $\epsilon_{E}$'),
        plt.Line2D([0], [0], marker='o', color='blue', markersize=15, linestyle='None', label=r'Incluyendo $\mathcal{R}$ pero no $\epsilon_{E}$'),
        plt.Line2D([0], [0], marker='o', color='green', markersize=15, linestyle='None', label=r'Incluyendo $\mathcal{R}$ y $\epsilon_{E}$'),
        plt.Line2D([0], [0], marker='s', color='Black', markersize=10, linestyle='None', label=r'$\mathcal{F}(q^2) = 1  $'),
        plt.Line2D([0], [0], marker='^', color='Black', markersize=10, linestyle='None', label=r'$\mathcal{F}(q^2) \neq 1$  ')
    ]
    ax20.grid(True, linestyle='--', alpha=0.8)
    ax20.set_xlim(Pe_range[2], Pe_range[-1])
    ax20.set_ylim(0,250)
    ax20.legend(handles=legend_elements, fontsize=12, handlelength=1.5, handletextpad=1.5)
    if save == True: 
        plt.savefig('comparacion eventos.png')
    else:
        plt.show()
    
'chi^2 con la aproximación final'  
if grafchieffi and Graf:
    fig19, ax19 = plt.subplots()
    contour_rec = ax19.contour(qu_rec, qe_rec, Z_rec, levels=[0, sigma], colors=['blue'], linestyles='--', linewidths=2)
    contour_E = ax19.contour(qu_rec, qe_rec, Z_E, levels=[0, sigma], colors=['red'], linestyles='-.', linewidths=2)
    contour_R = ax19.contour(qu_rec, qe_rec, Z_R, levels=[0, sigma], colors=['purple'], linestyles=':', linewidths=2)
    contour_RE = ax19.contour(qu_rec, qe_rec, Z_RE, levels=[0, sigma], colors=['green'], linestyles='-', linewidths=2)
    scatter_rec = ax19.scatter(qu_rec[x_min_rec,y_min_rec], qe_rec[x_min_rec,y_min_rec], color='blue', marker='+', s=50)
    scatter_E = ax19.scatter(qu_rec[x_min_E,y_min_E], qe_rec[x_min_E,y_min_E], color='red', marker='+', s=50)
    scatter_R = ax19.scatter(qu_rec[x_min_R,y_min_R], qe_rec[x_min_rec,y_min_R], color='purple', marker='+', s=50)
    scatter_RE = ax19.scatter(qu_rec[x_min_RE,y_min_RE], qe_rec[x_min_RE,y_min_RE], color='green', marker='+', s=50)
    scatter_sm = ax19.scatter(Q_2_SM, Q_2_SM, color='yellow', marker='+', s=50, label='Valor modelo estandar')
    ax19.set_xlabel(r'$\tilde{Q}_{\mu}^{2}$', fontsize=20)
    ax19.set_ylabel(r'$\tilde{Q}_{e}^{2}$', fontsize=20)
    ax19.set_title(r'$\chi^2 \left( \tilde{Q}_{\mu}^{2},\tilde{Q}_{e}^{2} \right)$ añadiendo $\mathcal{R}(T_{ee}^{rec},T_{ee}(T_{nr}))$ y $\epsilon(T_{ee}^{rec})$ ', fontsize=22)        
    ax19.axis([extremos_mu_rec[0],extremos_mu_rec[1],extremos_ee_rec[0],extremos_ee_rec[1]])
    ax19.grid(True, linestyle='--', alpha=0.8)
    ax19.legend()
    plt.show()

    legend_elements = [
        plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=2, label= r'Sin incluir $\mathcal{R}$ ni $\epsilon$'),
        plt.Line2D([0], [0], color='red', linestyle='-.', linewidth=2, label= r'Sin incluir $\mathcal{R}$ pero si $\epsilon$' ),
        plt.Line2D([0], [0], color='purple', linestyle=':', linewidth=2, label=r'Incluyendo $\mathcal{R}$ pero no $\epsilon$' ),
        plt.Line2D([0], [0], color='green', linestyle='-', linewidth=2, label=r'Incluyendo $\mathcal{R}$ y $\epsilon$'),
        plt.Line2D([0], [0], marker='+', color='black', markersize=5,linestyle='None', label=r'Valores minimos de $\chi^2$ '),
        plt.Line2D([0], [0], marker='+', color='yellow', markersize=5,linestyle='None', label=r'Valor modelo estándar para $\tilde{Q}_{W}$ ')]
    ax19.legend(handles=legend_elements, loc='best', fontsize=20)
    if save == True: 
        plt.savefig('chi2 final muchas lineas.png')
    else:
        plt.show()

'chi^2 con la aproximación final, menos lineas'
if grafchieffi and Graf:
    fig3, ax3 = plt.subplots(dpi = 190)
    contour_rec_f = ax3.contour(qu_rec, qe_rec, Z_rec_f, levels=[0, sigma], colors=['blue'], linestyles='-', linewidths=1.5)   # Sin nada
    contour_rec = ax3.contour(qu_rec, qe_rec, Z_rec, levels=[0, sigma], colors=['red'], linestyles=':', linewidths=1.5)      # Con FF
    contour_R_f = ax3.contour(qu_rec, qe_rec, Z_R_f, levels=[0, sigma], colors=['orange'], linestyles='-.', linewidths=1.5)
    contour_RE = ax3.contour(qu_rec, qe_rec, Z_RE, levels=[0, sigma], colors=['purple'], linestyles='--', linewidths=1.5)    #Con todo 
    scatter_sm = ax3.scatter(Q_2_SM, Q_2_SM, color='green', marker='+', s=50, label='Valor modelo estandar')      
    ax3.set_xlabel(r'$\tilde{Q}_{\mu}^{2}$', fontsize=16)
    ax3.set_ylabel(r'$\tilde{Q}_{e}^{2}$', fontsize=16)
    ax3.set_title(r'$\chi^2 \left( \tilde{Q}_{\mu}^{2},\tilde{Q}_{e}^{2} \right)$ con bins energéticos', fontsize=18)        
    ax3.axis([extremos_mu_rec[0],extremos_mu_rec[1],extremos_ee_rec[0],extremos_ee_rec[1]])
    ax3.grid(True, linestyle='--', alpha=0.8)
    ax3.legend(fontsize=12)
    plt.show()    
    legend_elements = [
        plt.Line2D([0], [0], color='blue', linestyle='-', linewidth=2, label= r'$\chi^2$ de figura 11'),
        plt.Line2D([0], [0], color='red', linestyle=':', linewidth=2, label= r'$\mathcal{F}(q^2) \neq  1 $' ),
        plt.Line2D([0], [0], color='orange', linestyle='-.', linewidth=2, label=r'Añadiendo $\mathcal{R}$'),
        plt.Line2D([0], [0], color='purple', linestyle='--', linewidth=2, label=r'Aproximación final'),
        plt.Line2D([0], [0], marker='+', color='green', markersize=15,linestyle='None', label=r'Valor del SM para $Q_{W}$ ')
        ]
    ax3.legend(handles=legend_elements, loc='best', fontsize=12)
    if save == True: 
        plt.savefig('chi2 final.png')
    else:
        plt.show()

'Marginalizacion de la funcion chi^2 con la aproximación final'
if grafmarefi:
    fig24, (ax241, ax242) = plt.subplots(2, 1, dpi=170)
    fig24.suptitle(r'Marginalización de $\chi^{2} (\tilde{Q}_{\mu}^{2},\tilde{Q}_{e}^{2})$ con bins energéticos' , fontsize=18)
    
    ax241.set_ylabel(r'$\chi^{2} (\tilde{Q}_{\mu}^{2})$', fontsize=16)  
    ax241.set_xlabel(r'$\tilde{Q}_{\mu}^{2}$', fontsize=16)  
    line242, = ax241.plot(q_array_mu_rec, Z_RE_mu, c='red', linewidth=1.5, label=r'$\chi^{2}$ con aproximación final')
    line243, = ax241.plot(q_array_mu_rec, Z_mu, c='blue', linewidth=1.5, label=r'$\chi^{2}$ de figura 12')
    line844 = ax241.axvline(x=Q_2_SM, color='black', linestyle='--',label='Valor SM')
    ax241.axvspan(q_array_mu_rec[int(indice_RE_mu[0,0])], q_array_mu_rec[int(indice_RE_mu[0,-1])], color='red', alpha=0.15, linewidth=2, label=r'1 $\sigma$ con aproximación final')
    ax241.axvspan(q_array_mu_rec[int(indice_mu[0,0])], q_array_mu_rec[int(indice_mu[0,-1])], color='blue', alpha=0.15, linewidth=2, label=r'1 $\sigma$ de figura 12')
    ax241.axis([-2000.,13000.,-np.max(Z_RE_mu)*0.01,np.max(Z_RE_mu)])
    ax241.grid(True, linestyle='--', alpha=0.8)
    ax241.legend(loc='upper center', fontsize=10)

    ax242.set_ylabel(r'$\chi^{2} (\tilde{Q}_{e}^{2})$', fontsize=16)  
    ax242.set_xlabel(r'$\tilde{Q}_{e}^{2}$', fontsize=16)   
    line242, = ax242.plot(q_array_ee_rec, Z_RE_ee, c='red', linewidth=1.5, label=r'$\chi^{2}$ con aproximación final')
    line242, = ax242.plot(q_array_ee_rec, Z_ee, c='blue', linewidth=1.5, label=r'$\chi^{2}$ de figura 12')
    line844 = ax242.axvline(x=Q_2_SM, color='black', linestyle='--',label='Valor SM')
    ax242.axvspan(q_array_ee_rec[int(indice_RE_ee[0,0])], q_array_ee_rec[int(indice_RE_ee[0,-1])], color='red', alpha=0.15, linewidth=2, label=r'1 $\sigma$ con aproximación final')
    ax242.axvspan(q_array_ee_rec[int(indice_ee[0,0])], q_array_ee_rec[int(indice_ee[0,-1])], color='blue', alpha=0.15, linewidth=2, label=r'1 $\sigma$ de figura 12')
    ax242.axis([-8800.,20000.,0,np.max(Z_RE_ee)])
    ax242.grid(True, linestyle='--', alpha=0.8)
    ax242.legend(loc='upper center', fontsize=10)
    if save == True: 
        plt.savefig('marginal final.png')
    else:
        plt.show()   
        
'Grafica del chi con la distribucion energetica y temporal'        
if grafchibi and Graf:
    fig6, ax6 = plt.subplots(dpi = 190)
    contour_rec_f = ax6.contour(qu_bi, qe_bi, Z_rec_f_bi, levels=[0, sigma], colors=['blue'], linestyles='-', linewidths=1.5)   # Sin nada
    contour_rec = ax6.contour(qu_bi, qe_bi, Z_rec_bi, levels=[0, sigma], colors=['red'], linestyles=':', linewidths=1.5)      # Con FF
    contour_R_f = ax6.contour(qu_bi, qe_bi, Z_R_f_bi, levels=[0, sigma], colors=['orange'], linestyles='-.', linewidths=1.5)
    contour_RE = ax6.contour(qu_bi, qe_bi, Z_RE_bi, levels=[0, sigma], colors=['purple'], linestyles='--', linewidths=1.5)    #Con todo 
    scatter_sm = ax6.scatter(Q_2_SM, Q_2_SM, color='green', marker='+', s=50, label='Valor modelo estandar')      
    ax6.set_xlabel(r'$\tilde{Q}_{\mu}^{2}$', fontsize=16)
    ax6.set_ylabel(r'$\tilde{Q}_{e}^{2}$', fontsize=16)
    ax6.set_title(r'$\chi^2 \left( \tilde{Q}_{\mu}^{2},\tilde{Q}_{e}^{2} \right)$ con bins energéticos y temporales', fontsize=18)        
    ax6.axis([extremos_mu_bi[0],extremos_mu_bi[1],extremos_ee_bi[0],extremos_ee_bi[1]])
    ax6.grid(True, linestyle='--', alpha=0.8)
    ax6.legend(fontsize=12)
    plt.show()    
    legend_elements = [
        plt.Line2D([0], [0], color='blue', linestyle='-', linewidth=2, label= r'Aproximación inicial'),
        plt.Line2D([0], [0], color='red', linestyle=':', linewidth=2, label= r'$\mathcal{F}(q^2) \neq  1 $' ),
        plt.Line2D([0], [0], color='orange', linestyle='-.', linewidth=2, label=r'Añadiendo $\mathcal{R}$'),
        plt.Line2D([0], [0], color='purple', linestyle='--', linewidth=2, label=r'Aproximación final'),
        plt.Line2D([0], [0], marker='+', color='green', markersize=15,linestyle='None', label=r'Valor del SM para $Q_{W}$ ')
        ]
    ax6.legend(handles=legend_elements, loc='best', fontsize=12)
    if save == True: 
        plt.savefig('chi2 bidi.png')
    else:
        plt.show()
        
'Marginalizamos chi con la distribucion energetica y temporal'           
if grafmarbi:
    fig1, (ax11, ax12) = plt.subplots(2, 1, dpi=190)
    fig1.suptitle(r'Marginalización de $\chi^{2} (\tilde{Q}_{\mu}^{2},\tilde{Q}_{e}^{2})$ con bins energéticos y temporales', fontsize=18)
    
    ax11.set_ylabel(r'$\chi^{2} (\tilde{Q}_{\mu}^{2})$', fontsize=16)  
    ax11.set_xlabel(r'$\tilde{Q}_{\mu}^{2}$', fontsize=16)  
    line11, = ax11.plot(q_array_mu_bi, Z_RE_mu_bi, c='red', linewidth=1.5, label=r'$\chi^{2}$ con aproximación final')
    line12, = ax11.plot(q_array_mu_bi, Z_mu_bi, c='blue', linewidth=1.5, label=r'$\chi^{2}$ con aproximación inicial')
    line13 = ax11.axvline(x=Q_2_SM, color='black', linestyle='--',label='Valor SM')
    ax11.axvspan(q_array_mu_bi[int(indice_RE_mu_bi[0,0])], q_array_mu_bi[int(indice_RE_mu_bi[0,-1])], color='red', alpha=0.15, linewidth=2, label=r'1 $\sigma$ con aproximación final')
    ax11.axvspan(q_array_mu_bi[int(indice_mu_bi[0,0])], q_array_mu_bi[int(indice_mu_bi[0,-1])], color='blue', alpha=0.15, linewidth=2, label=r'1 $\sigma$ con aproximación inicial')
    ax11.axis([extremos_mu_bi[0],extremos_mu_bi[1] ,-np.max(Z_RE_mu_bi)*0.01,np.max(Z_RE_mu_bi)])
    ax11.grid(True, linestyle='--', alpha=0.8)
    ax11.legend(loc='upper center', fontsize=10)

    ax12.set_ylabel(r'$\chi^{2} (\tilde{Q}_{e}^{2})$', fontsize=16)  
    ax12.set_xlabel(r'$\tilde{Q}_{e}^{2}$', fontsize=16)   
    line21, = ax12.plot(q_array_ee_bi, Z_RE_ee_bi, c='red', linewidth=1.5, label=r'$\chi^{2}$ con aproximación final')
    line22, = ax12.plot(q_array_ee_bi, Z_ee_bi, c='blue', linewidth=1.5, label=r'$\chi^{2}$ con aproximación inicial')
    line23 = ax12.axvline(x=Q_2_SM, color='black', linestyle='--',label='Valor SM')
    ax12.axvspan(q_array_ee_bi[int(indice_RE_ee_bi[0,0])], q_array_ee_bi[int(indice_RE_ee_bi[0,-1])], color='red', alpha=0.15, linewidth=2, label=r'1 $\sigma$ con aproximación final')
    ax12.axvspan(q_array_ee_bi[int(indice_ee_bi[0,0])], q_array_ee_bi[int(indice_ee_bi[0,-1])], color='blue', alpha=0.15, linewidth=2, label=r'1 $\sigma$ con aproximación inicial')
    ax12.axis([extremos_ee_bi[0],extremos_ee_bi[1] ,0,np.max(Z_RE_ee_bi)])
    ax12.grid(True, linestyle='--', alpha=0.8)
    ax12.legend(loc='upper center', fontsize=10)
    if save == True: 
        plt.savefig('marginal bi.png')
    else:
        plt.show()    
        
'Grafica del chi comaparando distribucion energetica y temporal'           
 if grafchicobi and Graf:
    fig7, ax7 = plt.subplots(dpi = 190) 
    contour_rec_com = ax7.contour(qu_rec, qe_rec, Z_rec_f_com, levels=[0, sigma], colors=['purple'], linestyles='-', linewidths=2)
    contour_RE_com = ax7.contour(qu_rec, qe_rec, Z_RE_com, levels=[0, sigma], colors=['purple'], linestyles='-.', linewidths=2)
    contour_rec_f_com = ax7.contour(qu_rec, qe_rec, Z_rec_f_bi_com, levels=[0, sigma], colors=['blue'], linestyles='-', linewidths=1.5) 
    contour_RE_com = ax7.contour(qu_rec, qe_rec, Z_RE_bi_com, levels=[0, sigma], colors=['blue'], linestyles='-.', linewidths=1.5)    
    scatter_sm = ax7.scatter(Q_2_SM, Q_2_SM, color='green', marker='+', s=50, label='Valor modelo estandar')      
    ax7.set_xlabel(r'$\tilde{Q}_{\mu}^{2}$', fontsize=16)
    ax7.set_ylabel(r'$\tilde{Q}_{e}^{2}$', fontsize=16)
    ax7.set_title(r' Variación de $\chi^2 \left( \tilde{Q}_{\mu}^{2},\tilde{Q}_{e}^{2} \right)$ con la inclusion de bins temporales', fontsize=18)        
    ax7.axis([-2500.,13500.,-10000.,22000.])
    ax7.grid(True, linestyle='--', alpha=0.8)
    ax7.legend(fontsize=12)  
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='blue', markersize=15, linestyle='None', label=r'Con bins temporales'),
        plt.Line2D([0], [0], marker='o', color='purple', markersize=15, linestyle='None', label=r'Sin bins temporales'),
        plt.Line2D([0], [0], color='black', linestyle='-', linewidth=2, label= r'Aproximación inicial'),
        plt.Line2D([0], [0], color='black', linestyle='-.', linewidth=2, label= r'Aproximación final')]
    ax7.legend(handles=legend_elements, loc='best', fontsize=12)
    if save == True: 
        plt.savefig('chi2 bicom.png')
    else:
        plt.show() 
#################################################################################################################################################
#################################################################################################################################################
'Imprimimos algunos valores de interes'
#################################################################################################################################################
#################################################################################################################################################
if N_eventos_naive :
    print("Numero de eventos total (1Q / 2Q) :")
    print(N_tot, '/' ,N_totq2,'\n')
    print("Numero de eventos prompt (1Q / 2Q) :")
    print(N_prompt, '/' ,N_promptq2,'\n' )
    print("Numero de eventos delayed (1Q / 2Q) :")
    print(N_tot, '/' ,N_delay_eq2,'\n' )
    print("Numero de eventos total (1Q / 2Q) :")
    print(N_delay_e + N_delay_u, '/' ,N_delay_eq2 + N_delay_uq2,'\n' )

    
if graffcorr:
    print("Matriz de covarianza:")
    print(cov_sqr,'\n')
    print("Matriz de covarianza inversa:")
    print(np.linalg.inv(cov_sqr),'\n')
    print("Matriz de covarianza diagonalizada:")
    print(cov_diag_sqr,'\n')
    print("Matriz que diagonaliza la matriz de covarianza:")
    print(trans_sqr,'\n')
    print("Algulo de rotacion")
    print(np.degrees(ang_sqr),'\n')
    print('Matriz de correlacion')
    print(corr_matrix,'\n')
    print('Matriz de correlacion elipse tumbada')
    print(corr_matrix_C,'\n')
    
if grafcorrefici:
    print("Matriz de covarianza:")
    print(cov_efi,'\n')
    print("Matriz de covarianza inversa:")
    print(np.linalg.inv(cov_efi),'\n')
    print("Matriz de covarianza diagonalizada:")
    print(cov_diag_efi,'\n')
    print("Matriz que diagonaliza la matriz de covarianza:")
    print(trans_efi,'\n')
    print("Algulo de rotacion")
    print(np.degrees(ang_efi),'\n')
    print('Matriz de correlacion')
    print(corr_matrix_efi,'\n')   

if grafcorbi:
    print("Matriz de covarianza:")
    print(cov_efi_bi,'\n')
    print("Matriz de covarianza inversa:")
    print(np.linalg.inv(cov_efi_bi),'\n')
    print("Matriz de covarianza diagonalizada:")
    print(cov_diag_efi_bi,'\n')
    print("Matriz que diagonaliza la matriz de covarianza:")
    print(trans_efi_bi,'\n')
    print("Algulo de rotacion")
    print(np.degrees(ang_efi_bi),'\n')
    print('Matriz de correlacion')
    print(corr_matrix_efi_bi,'\n')   
     
if graf_time: 
    print(r'Coeficientes g prompt:')
    print(g_p)
    print(r'Coeficientes g delay:')
    print(g_d)

if grafdatacomp:
    print("Numero de eventos naive: ")
    print(np.round(np.sum(N_bins_rec_f)),'\n')
    print("Numero de eventos con F(q^2): ")
    print(np.round(np.sum(N_bins_rec[1:])+ N_bins_rec_f[0]),'\n')
    print("Aplicando la funcion R: ")
    print(np.round(np.sum(N_bins_R_f[1:])+N_bins_rec_f[0]),'\n')
    print("Aplicando la funcion R y F : ")
    print(np.round(np.sum(N_bins_R[1:])+N_bins_rec_f[0]),'\n')
    print("Aplicando todo menos F : ")
    print(np.round(np.sum(N_bins_RE_f)),'\n')
    print("Aproximación final: ")
    print(np.round(np.sum(N_bins_RE)),'\n')
    
if valsigma: 
    confidence_level = stats.chi2.cdf(sigma, df=2)
    print(round(confidence_level*100,2))
    print(stats.chi2.ppf(0.68, df=2))
    
#################################################################################################################################################
#################################################################################################################################################
'Devolvemos tiempo de ejecucion'
#################################################################################################################################################
#################################################################################################################################################

if times == True: 
    end_time = time.time()
    execution_time = end_time - start_time
    print("Tiempo de ejecución:", execution_time, "segundos")
    
    
