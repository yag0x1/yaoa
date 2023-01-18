import scipy.constants as sc
import numpy as np
from scipy import signal


#Vetor direção para a ULA de 4 elementos
def Sv_ULA(phi,C,beta):
  phi = phi*np.pi/180

  Svk = np.array(np.zeros([1,4])+1j*np.zeros([1,4]))
  for a in range(4):
    Svk[0,a] = np.exp(1j*beta*(C[a,0]* np.cos(phi)))
  return Svk

#PMU ULA de 4 elementos
def PMU_ULA(phi,Cpl,C,beta,En):
  max=1e4
  if 1/(np.linalg.norm(np.transpose(np.conj((Cpl*np.matrix(Sv_ULA(phi,C,beta).T)).T)*En*np.identity(3))))**2 == 0:
    pmu = 10**37
  if 1/(np.linalg.norm(np.transpose(np.conj((Cpl*np.matrix(Sv_ULA(phi,C,beta).T)).T)*En*np.identity(3))))**2 > max:
    pmu=max
  else:
    pmu = 1/(np.linalg.norm(np.transpose(np.conj((Cpl*np.matrix(Sv_ULA(phi,C,beta).T)).T)*En*np.identity(3))))**2
  return pmu

#MUSIC para a ULA de 4 elementos
def dmusic_ULA(x_iq,f):
  VT=x_iq
  
  s = 0.32*(sc.speed_of_light/2.442e9) # espaçamento entre os elementos do hfss
  Z = np.array([[50 + 0j,0 + 0j,0 + 0j,0 + 0j],
              [0 + 0j,0 + 50j,0 + 0j,0 + 0j],
              [0 + 0j,0 + 0j,0 + 50j,0 + 0j],
              [0 + 0j,0 + 0j,0 + 0j,50 + 0j]])

  NSmpl = VT.shape[1]                #número de amostras
  ZT = 50  #impedancia na porta das antenas
  K=4      #numero de elementos
  lambd = sc.speed_of_light/(f*1e9) #compriemto de onda
  beta = 2*np.pi/lambd
  C = np.array([[0],[s],[2*s],[3*s]]) #posição dos dipolos

  #Caculo da matriz de acoplamento do array
  Cpl = (ZT+Z)*np.linalg.inv((Z+ZT*np.identity(K)))

  Nsg,NS = VT.shape   #NS-Numero de colunas, Nsg-Numero de linhas

  # Matriz de covariância
  Aa = np.matrix(VT)
  Cc = np.zeros([4,4])+1j*np.zeros([4,4])
  for x in range(NS):
    a = Aa[:,x]*Aa[:,x].getH()
    Cc = Cc+a

  Cov = Cc/NS

  # Calculo dos autovalores
  autovalores,autovetores= np.linalg.eig(np.matrix(Cov))
  
  U = autovetores
  Proj = np.conj(np.transpose(U))*VT

  En = U[:,1:4] #Sinais que são ruido no caso do BLE apenas um é o sinal de interesse os outros é ruido
  
  pmu_v = np.zeros([1,1801])
  phi_n = np.matrix(np.linspace(0,180,1801))
  for x in range(1801):
    pmu_v[0,x] = PMU_ULA(phi_n[0,x],Cpl,C,beta,En)

  ang_cal_string = pmu_v[0,:]/np.max(pmu_v[0,:])

  aaa=np.where(pmu_v[0,:]/np.max(pmu_v[0,:]) == 1)

  ang_cal = np.sum(np.array(aaa)*180/1801)/np.size(np.array(aaa))

  return ang_cal, ang_cal_string
  

#função que adiciona ruido a um sinal de entrada
def noise_add(x_v,SNR):

  sig_pw_db = 10 * np.log10(np.mean(abs(x_v) ** 2))
  r_pw_db = sig_pw_db - SNR

  r_pw_w = 10 ** (r_pw_db / 10)
  r_meia = 0
  n_v = np.random.normal(r_meia, np.sqrt(r_pw_w), len(x_v))

  y_v = x_v + n_v
  return y_v

#retorna uma matriz de ruido AWGN
def noise_gen(Pwsig_dB,SNR,N,N_samp):
  Pnoise = 10**((np.max(Pwsig_dB)-SNR)/10.0) 
  noise = np.sqrt(Pnoise/2)*(np.random.randn(N, N_samp) +1j*np.random.randn(N, N_samp)); #ruido não correlacionado
  return noise 
