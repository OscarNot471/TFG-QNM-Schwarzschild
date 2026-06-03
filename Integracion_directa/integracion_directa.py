import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw
from scipy.integrate import solve_ivp
from scipy.optimize import root


"""
Cambios de coordenadas
"""

def Calc_r_tort(r , M):
    #Cambio r -> r*
    return r + 2*M * np.log(r /(2*M) - 1)


def Calc_r(r_tort):
    #Cambio r* -> r
    z = np.exp(r_tort/(2*M) - 1)
    r = 2*M*(lambertw(z, 0).real + 1)
    return r


"""
Funciones útiles (f, f')
"""

def f(r, M):
    return 1 - 2*M/r

def df(r, M):
    return 2*M/r**2


"""
Implementación de las series
"""

def Calc_a_coefs(a_len, w, l, M):
    a = np.zeros(a_len, dtype = complex)
    
    a[0] = 1.0 + 0j
    
    a[1] = (3 - l*(l+1)) * a[0] / (1 - 4j*M*w)
    
    for n in range(2, a_len):
        Sum = 0
        for m in range(0, n):
            Sum += (-1)**m * a[n-m]
        a[n] = - (a[n-1] * ((n-1)*(n-2) - (8j*M*w*n) - l*(l+1)) - a[n-2] * (4j*M*w * (n-2)) + 3*Sum) / (n * (n - 4j*M*w))
    
    return a

def Calc_b_coefs(b_len, w, l, M):
    b = np.zeros(b_len, dtype = complex)
    
    b[0] = 1.0 + 0j
    
    b[1] = b[0] * (1j*l*(l+1)/(4*M*w))
    
    b[2] = (b[1] * (2 - l*(l+1)) + 3*b[0])/(8*M*w*1j)
    
    for n in range(3, b_len):
        b[n] = (b[n-1] * ((n-1)*n - l*(l+1)) + b[n-2] * (4 - (n-1)**2)) / (4j*M*w*n)
    
    return b


def Construir_S(r, w, a_coefs):
    x = (r - 2*M)/(2*M)
    S = 0.0j
    for n in range(len(a_coefs)):
        S += a_coefs[n] * x**n
    return S

def Construir_dS(r, w, a_coefs):
    x = (r - 2*M)/(2*M)
    S = 0.0j
    for n in range(1, len(a_coefs)):
        S += n * a_coefs[n] * x**(n-1)
    return S

def Construir_U(r, w, b_coefs):
    y = 2*M/r
    U = 0.0j
    for n in range(len(b_coefs)):
        U += b_coefs[n] * y**n
    return U

def Construir_dU(r, w, b_coefs):
    y = 2*M/r
    U = 0.0j
    for n in range(1, len(b_coefs)):
        U += n * b_coefs[n] * y**(n-1)
    return U

""" Implementamos la ecuación de Regge-Wheeler """

def Calc_V_axial(r):
    return (1 - 2*M/r) * (l*(l+1)/r**2 - 6*M/r**3)

def R_W(r_tort, Y, w):
        
    u = Y[0]
    v = Y[1]
    
    r = Calc_r(r_tort)
    V_axial = Calc_V_axial(r)
    
    du = v
    dv = -(w**2 - V_axial)*u

    return np.array([du, dv], dtype=complex)


""" Funciones que aproximan f+ f- y sus derivadas en su extremo """

def Calc_Y_h(a_len, w, l, M, r_h, r_tort_h):
    a_coefs = Calc_a_coefs(a_len, w, l, M)
    
    S = Construir_S(r_h, w, a_coefs)
    dS = Construir_dS(r_h, w, a_coefs)
    
    f_menos = np.exp(-1j*w*r_tort_h) * S
    df_menos = np.exp(-1j*w*r_tort_h) * (-1j*w*S + (f(r_h, M)/(2*M)*dS))
    
    Y_h = np.array([f_menos, df_menos], dtype= complex)
    return Y_h

def Calc_Y_inf(b_len, w, l, M, r_inf, r_tort_inf):
    b_coefs = Calc_b_coefs(b_len, w, l, M)
    
    U = Construir_U(r_inf, w, b_coefs)
    dU = Construir_dU(r_inf, w, b_coefs)
    
    f_mas = np.exp(1j*w*r_tort_inf) * U
    df_mas = np.exp(1j*w*r_tort_inf) * (1j*w*U - (2*M/r_inf**2)*f(r_inf, M)*dU)
    
    Y_inf = np.array([f_mas, df_mas], dtype= complex)
    return Y_inf

""" Integración desde el horizonte hasta el punto de matching """
def integrar_h(r_tort_min, r_tort_match, Y_h, w):
    intervalo_h = np.array([r_tort_min, r_tort_match])
    paso = abs(r_tort[1] - r_tort[0])
    r_eval_h = np.linspace(r_tort_min, r_tort_match, N_match + 1)
    
    sol_h = solve_ivp(R_W, intervalo_h, Y_h, args=(w,), method="DOP853", t_eval=r_eval_h, max_step=paso, rtol = 1e-10, atol = 1e-10)
    
    Psi_h_match = sol_h.y[0, -1]
    dPsi_h_match = sol_h.y[1, -1]
    return (Psi_h_match, dPsi_h_match)

""" Integración desde el infinito hasta el punto de matching """
def integrar_inf(r_tort_max, r_tort_match, Y_inf, w):
    intervalo_inf = np.array([r_tort_max, r_tort_match])
    paso = abs(r_tort[1] - r_tort[0])
    r_eval_inf = np.linspace(r_tort_max, r_tort_match, N - N_match)
    
    sol_inf = solve_ivp(R_W, intervalo_inf, Y_inf, args=(w,), method="DOP853", t_eval=r_eval_inf, max_step=paso, rtol = 1e-10, atol = 1e-10)
    
    Psi_inf_match = sol_inf.y[0, -1]
    dPsi_inf_match = sol_inf.y[1, -1]
    return (Psi_inf_match, dPsi_inf_match)


""" Implementamos el Wronskiano (la función del sistema F(x) = 0) """

def F(w_vec):

    w = w_vec[0] + 1j*w_vec[1]
    
    Y_h = Calc_Y_h(a_len, w, l, M, r_h, r_tort_h)
    Y_inf = Calc_Y_inf(b_len, w, l, M, r_inf, r_tort_inf)
    
    Psi_h_match, dPsi_h_match = integrar_h(r_tort_min, r_tort_match, Y_h, w)
    Psi_inf_match, dPsi_inf_match = integrar_inf(r_tort_max, r_tort_match, Y_inf, w)
    
    W = Psi_h_match * dPsi_inf_match - Psi_inf_match * dPsi_h_match
    
    return np.array([W.real, W.imag])

""" Main """

#Unidades geométricas
c = 1
G = 1

M = 1

#Parámetros del Modo
l = 2
w_guess = 0.37 - 0.09j

#Número de coeficientes de las series
a_len = 50
b_len = 50


#Mallados radiales
N = 500

r_tort_min = -100*M 
r_tort_max =  100*M

r_tort = np.linspace(r_tort_min, r_tort_max, N)

r = Calc_r(r_tort)

#Radios donde centramos las serie
r_tort_h = r_tort_min
r_tort_inf = r_tort_max

r_h = Calc_r(r_tort_h)
r_inf = Calc_r(r_tort_inf)

#Punto de matching
N_match = 250
r_tort_match = r_tort[N_match]

""" Calculamos las raíces de la función F(x) = 0 """

x0 = np.array([w_guess.real, w_guess.imag])
sol_root = root(F, x0)

print("success =", sol_root.success)
print("message =", sol_root.message)

w_sol = sol_root.x[0] + 1j*sol_root.x[1]

print("omega encontrado =", w_sol)
print("residuo final =", F(sol_root.x))




""" Reconstrucción de las soluciones radiales para omega encontrada """

# Construimos las condiciones iniciales en los extremos usando la frecuencia encontrada
Y_h = Calc_Y_h(a_len, w_sol, l, M, r_h, r_tort_h)
Y_inf = Calc_Y_inf(b_len, w_sol, l, M, r_inf, r_tort_inf)

# Mallados para guardar la solución completa
N_plot = 1000

r_eval_h = np.linspace(r_tort_min, r_tort_match, N_plot)
r_eval_inf = np.linspace(r_tort_max, r_tort_match, N_plot)

# Integramos desde el horizonte guardando todos los puntos
sol_h_full = solve_ivp(
    R_W,
    (r_tort_min, r_tort_match),
    Y_h,
    args=(w_sol,),
    method="DOP853",
    t_eval=r_eval_h,
    rtol=1e-10,
    atol=1e-10
)

# Integramos desde el infinito guardando todos los puntos
sol_inf_full = solve_ivp(
    R_W,
    (r_tort_max, r_tort_match),
    Y_inf,
    args=(w_sol,),
    method="DOP853",
    t_eval=r_eval_inf,
    rtol=1e-10,
    atol=1e-10
)

# Extraemos coordenada tortuga y soluciones
r_tort_h_plot = sol_h_full.t
f_menos_plot = sol_h_full.y[0]

r_tort_inf_plot = sol_inf_full.t
f_mas_plot = sol_inf_full.y[0]


# Factor complejo para que f_+ coincida con f_- en el punto de matching
C = f_menos_plot[-1] / f_mas_plot[-1]

# Reescalamos f_+
f_mas_matched = C * f_mas_plot

# Normalizamos ambas ramas para que Q(r_match) = 1
f_menos_norm = f_menos_plot / f_menos_plot[-1]
f_mas_norm = f_mas_matched / f_menos_plot[-1]



# La rama desde infinito está en orden decreciente de r_*
# La ordenamos de menor a mayor para poder unirla con la rama del horizonte
idx_inf = np.argsort(r_tort_inf_plot)

r_inf_sorted = r_tort_inf_plot[idx_inf]
f_mas_sorted = f_mas_norm[idx_inf]

# Unimos: rama desde horizonte hasta matching + rama desde matching hasta infinito
# Quitamos un punto repetido cerca del matching
r_Q = np.concatenate([r_tort_h_plot, r_inf_sorted[1:]])
Q_plot = np.concatenate([f_menos_norm, f_mas_sorted[1:]])



# Gráficas de f_- y f_+ normalizadas
plt.figure()
plt.plot(r_tort_h_plot, np.abs(f_menos_norm), "-", label=r"$f_-$ desde el horizonte")
plt.plot(r_tort_inf_plot, np.abs(f_mas_norm), "-", label=r"$f_+$ desde el infinito")
plt.axvline(r_tort_match, color="k", linestyle=":")
plt.xlabel(r"$r_*$")
plt.ylabel(r"$|f_{\pm}|$")
plt.legend()
plt.title(r"Módulo de las soluciones normalizadas")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(r_tort_h_plot, f_menos_norm.real, "-", label=r"$f_-$ desde el horizonte")
plt.plot(r_tort_inf_plot, f_mas_norm.real, "-", label=r"$f_+$ desde el infinito")
plt.axvline(r_tort_match, color="k", linestyle=":")
plt.xlabel(r"$r_*$")
plt.ylabel(r"$\mathrm{Re}(f_{\pm})$")
plt.legend()
plt.title(r"Parte real de las soluciones normalizadas")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(r_tort_h_plot, f_menos_norm.imag, "-", label=r"$f_-$ desde el horizonte")
plt.plot(r_tort_inf_plot, f_mas_norm.imag, "-", label=r"$f_+$ desde el infinito")
plt.axvline(r_tort_match, color="k", linestyle=":")
plt.xlabel(r"$r_*$")
plt.ylabel(r"$\mathrm{Im}(f_{\pm})$")
plt.legend()
plt.title(r"Parte imaginaria de las soluciones normalizadas")
plt.tight_layout()
plt.show()


# Gráficas de la solución radial reconstruida Q
plt.figure()
plt.plot(r_Q, np.abs(Q_plot), "-")
plt.axvline(r_tort_match, color="k", linestyle=":")
plt.xlabel(r"$r_*$")
plt.ylabel(r"$|Q(r_*)|$")
plt.title(r"Módulo de la solución radial reconstruida")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(r_Q, Q_plot.real, label=r"$\mathrm{Re}(Q)$")
plt.plot(r_Q, Q_plot.imag, label=r"$\mathrm{Im}(Q)$")
plt.axvline(r_tort_match, color="k", linestyle=":")
plt.xlabel(r"$r_*$")
plt.ylabel(r"$Q(r_*)$")
plt.title(r"Solución radial reconstruida $Q(r_*)$")
plt.legend()
plt.tight_layout()
plt.show()

#Módulo de Q en escala logarítmica
plt.figure()
plt.semilogy(r_Q, np.abs(Q_plot), "-")
plt.axvline(r_tort_match, color="k", linestyle=":")
plt.xlabel(r"$r_*$")
plt.ylabel(r"$|Q(r_*)|$")
plt.title(r"Módulo de $Q(r_*)$ en escala logarítmica")
plt.tight_layout()
plt.show()
