import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt


""" Unidades y parámetros """

G = 1
c = 1
M = 0.5          # Unidades de Leaver: 2M = 1

epsilon = 3      # Perturbaciones gravitacionales
N = 5000         # Truncamiento de la fracción continua


""" Semillas para los 60 modos """

n_l2_semillas = np.arange(1, 61)
n_l3_semillas = np.arange(1, 61)


# l = 2

semillas_l2 = np.array([
    0.7473433688360837 - 0.1779246313778714j,
    0.6934219937583269 - 0.5478297505824696j,
    0.6021069092247329 - 0.9565539664461438j,
    0.5030099243711813 - 1.4102964048669908j,
    0.4150291596261310 - 1.8936897817327030j,
    0.3385988061860883 - 2.3912161082716930j,
    0.2665046804895833 - 2.8958212523130817j,
    0.1856446915056352 - 3.4076823755109080j,
    0.0000000000000000 - 3.9845075240609350j,
    0.1265113495696696 - 4.6052814672543270j,
    0.1531055606701001 - 5.1216586759328390j,
    0.1651932477670230 - 5.6308880969614000j,
    0.1714547598058120 - 6.1373845523059750j,
    0.1747870046005531 - 6.6424667303970210j,
    0.1764860164952248 - 7.1466442858850550j,
    0.1771925346736894 - 7.6502098051835940j,
    0.1772797706944637 - 8.1533370871180550j,
    0.1769453784314204 - 8.6561205730957940j,
    0.1763669551869236 - 9.1585690414054430j,
    0.1756655467663188 - 9.6608876741837990j,
    0.1747763321459832 - 10.162886780010950j,
    0.1738219626743588 - 10.664888794906036j,
    0.1729535376668657 - 11.166680152486165j,
    0.1720181810653184 - 11.668264575788797j,
    0.1710340230928441 - 12.169754885511914j,
    0.1700664608712970 - 12.671191652005815j,
    0.1691284805446951 - 13.172613566414713j,
    0.1681266041576552 - 13.674051643732994j,
    0.1669225848457954 - 14.175325936741055j,
    0.1658756840951497 - 14.676116845952263j,
    0.1654606173689608 - 15.177192836175502j,
    0.1642564680322380 - 15.678769953081101j,
    0.1632982127406648 - 16.179010006274059j,
    0.1628496180094738 - 16.680739242454308j,
    0.1615267428365868 - 17.180788280865912j,
    0.1610585192910165 - 17.682710928556791j,
    0.1604982220364013 - 18.182400883586819j,
    0.1586998754448176 - 18.683929811048063j,
    0.1591811285981317 - 19.185147423049564j,
    0.1585229700107790 - 19.684633224440447j,
    0.1565227716249322 - 20.185356515026221j,
    0.1557432921512140 - 20.687359937552007j,
    0.1561968789364001 - 21.188440104555514j,
    0.1563513978853702 - 21.688460257918589j,
    0.1558381077276884 - 22.188233227483455j,
    0.1549280143342347 - 22.688197148763617j,
    0.1539475309396608 - 23.188416608051917j,
    0.1531091676049961 - 23.688792809077153j,
    0.1525198731277561 - 24.189228644517847j,
    0.1522271796542203 - 24.689722767682884j,
    0.1522133277592014 - 25.190399450364680j,
    0.1523208881360649 - 25.691460717800855j,
    0.1521806924283813 - 26.193037098726251j,
    0.1512395237029935 - 26.694949445222221j,
    0.1490116696991000 - 27.196374838133686j,
    0.1461382700917997 - 27.695599090017488j,
    0.1461605195263828 - 28.192736744181349j,
    0.1491031344479102 - 28.692893185172757j,
    0.1498776985210142 - 29.196503414080425j,
    0.1460632696645224 - 29.699170425285743j
])


# l = 3

semillas_l3 = np.array([
    1.1988865768749801 - 0.1854060958898952j,
    1.1652876060665989 - 0.5625962268700880j,
    1.1033698015569027 - 0.9581855019339245j,
    1.0239238221166747 - 1.3806741919384782j,
    0.9403480116303105 - 1.8312987850101934j,
    0.8627729572843076 - 2.3043027242818110j,
    0.7953190483515147 - 2.7918244854451832j,
    0.7379845517794562 - 3.2876890567135330j,
    0.6892366371902323 - 3.7880656083859252j,
    0.6473662632026924 - 4.2907978995346410j,
    0.6109218038621906 - 4.7947091006032260j,
    0.5787681902376183 - 5.2991592113172890j,
    0.5500387921677476 - 5.8037991423876360j,
    0.5240723189739591 - 6.3084389558272990j,
    0.5003594309231401 - 6.8129771848334600j,
    0.4785024409936133 - 7.3173623260239480j,
    0.4581860051911424 - 7.8215714413999190j,
    0.4391559776026426 - 8.3255980380698170j,
    0.4212040773261938 - 8.8294450942745680j,
    0.4041566354516268 - 9.3331210054939380j,
    0.3878661895892380 - 9.8366372434158790j,
    0.3722050809940972 - 10.340006967518642j,
    0.3570603542416842 - 10.843244412891170j,
    0.3423298897726823 - 11.346364539756641j,
    0.3279187811482202 - 11.849382817058981j,
    0.3137360919060446 - 12.352315637365010j,
    0.2996918379392039 - 12.855180982533536j,
    0.2856930614510763 - 13.357999184446611j,
    0.2716384658097152 - 13.860791599912321j,
    0.2574207406719886 - 14.363574888366342j,
    0.2429441086624882 - 14.866387469386964j,
    0.2280249709090914 - 15.369346220236165j,
    0.2123621466903136 - 15.872353405871225j,
    0.1961454158603767 - 16.375584801166994j,
    0.1781244443494136 - 16.879484879899987j,
    0.1594542123618998 - 17.383244204686136j,
    0.1360123671591729 - 17.889453362898831j,
    0.1139611523461758 - 18.396194418881727j,
    0.0725157116186819 - 18.904030618424894j,
    0.0490567223033677 - 19.434961306689665j,
    0.0000000000000000 - 20.026258559704459j,
    0.0203526333813687 - 20.585675763885057j,
    0.0705533654102144 - 21.087412266285195j,
    0.0860981199041229 - 21.591529944536088j,
    0.0949683605131437 - 22.100211724575384j,
    0.1004451268248110 - 22.355214572204270j,

    0.1059218931364788 - 22.610217419833159j,
    0.1172160892680944 - 23.116124649286103j,
    0.1261308217476386 - 23.619585421973266j,
    0.1329441869718519 - 24.122120049755676j,
    0.1381887811501266 - 24.624213875210387j,
    0.1422473543326498 - 25.125912411071237j,
    0.1454903889445647 - 25.627049011357990j,
    0.1485122431524367 - 26.127482477836036j,
    0.1520724245292795 - 26.627616444071514j,
    0.1562470522343025 - 27.128524645268538j,
    0.1600091862901805 - 27.630870080701740j,
    0.1620651607732346 - 28.134140512870964j,
    0.1617973734384036 - 28.636558913878883j,
    0.1613219750980716 - 29.135711811261579j
])


""" Funciones para calcular los coefs. de la recurrencia """

def Calc_alpha(w, N):
    p = -1j * w
    alpha = np.zeros(N, dtype=complex)

    for k in range(N):
        alpha[k] = k**2 + (2*p + 2)*k + 2*p + 1

    return alpha


def Calc_beta(w, N, l):
    p = -1j * w
    beta = np.zeros(N, dtype=complex)

    for k in range(N):
        beta[k] = -(
            2*k**2
            + (8*p + 2)*k
            + 8*p**2
            + 4*p
            + l*(l + 1)
            - epsilon
        )

    return beta


def Calc_gamma(w, N):
    p = -1j * w
    gamma = np.zeros(N, dtype=complex)

    for k in range(N):
        gamma[k] = k**2 + 4*p*k + 4*p**2 - epsilon - 1

    return gamma


""" Implementamos la ecuación característica de Leaver """
#n_inversion = 0 -> fracción continua original.
#n_inversion = 1 -> primera inversión.
#n_inversion = 2 -> segunda inversión.
#...

def F(x, n_inversion, l):
    w = x[0] + 1j*x[1]

    alpha = Calc_alpha(w, N)
    beta = Calc_beta(w, N, l)
    gamma = Calc_gamma(w, N)

    if n_inversion == 0:

        frac = beta[N - 1]

        for i in range(N - 2, 0, -1):
            frac = beta[i] - alpha[i] * gamma[i + 1] / frac

        F_complex = beta[0] - alpha[0] * gamma[1] / frac

    else:

        frac_izda = beta[0]

        for i in range(1, n_inversion + 1):
            frac_izda = beta[i] - alpha[i - 1] * gamma[i] / frac_izda

        L = frac_izda

        frac_dcha = beta[N - 1]

        for i in range(N - 2, n_inversion, -1):
            frac_dcha = beta[i] - alpha[i] * gamma[i + 1] / frac_dcha

        R = alpha[n_inversion] * gamma[n_inversion + 1] / frac_dcha

        F_complex = L - R

    return np.array([F_complex.real, F_complex.imag])


""" Implementamos el algoritmo de búsqueda de raíces """

def Calc_w(n_Leaver, omega_guess, l):
    n_inversion = n_Leaver - 1

    x0 = np.array([
        omega_guess.real,
        omega_guess.imag
    ])

    sol = root(
        F,
        x0,
        args=(n_inversion, l),
        method="hybr",
        tol=1e-12,
        options={
            "maxfev": 20000,
            "xtol": 1e-12
        }
    )

    omega = sol.x[0] + 1j*sol.x[1]

    if omega.real < 0:
        omega = -omega.real + 1j*omega.imag

    return omega, sol


""" Implementamos el algoritmo de cálculo sistemático de todos los modos """

def calcular_familia_directa(l, n_semillas, semillas):
    frecuencias = []
    residuos = []
    convergencias = []

    print("\n====================================================")
    print(f"CALCULANDO FAMILIA COMPLETA PARA l = {l}")
    print("====================================================")

    for n, guess in zip(n_semillas, semillas):

        omega, sol = Calc_w(n, guess, l)

        residuo = np.linalg.norm(sol.fun)

        frecuencias.append(omega)
        residuos.append(residuo)
        convergencias.append(sol.success)

        print("-----------------------------------")
        print(f"l = {l}, n Leaver = {n}")
        print(f"inversión = {n - 1}")
        print("guess =", guess)
        print("omega_Leaver =", omega)
        print("success =", sol.success)
        print("|residuo| =", residuo)

    return (
        np.array(n_semillas),
        np.array(frecuencias),
        np.array(residuos),
        np.array(convergencias)
    )


""" Main """

n_l2, frec_l2, res_l2, conv_l2 = calcular_familia_directa(
    l=2,
    n_semillas=n_l2_semillas,
    semillas=semillas_l2
)

n_l3, frec_l3, res_l3, conv_l3 = calcular_familia_directa(
    l=3,
    n_semillas=n_l3_semillas,
    semillas=semillas_l3
)


""" Guardamos los resultados """

datos_l2 = np.column_stack([
    n_l2,
    frec_l2.real,
    frec_l2.imag,
    res_l2,
    conv_l2.astype(int)
])

datos_l3 = np.column_stack([
    n_l3,
    frec_l3.real,
    frec_l3.imag,
    res_l3,
    conv_l3.astype(int)
])

cabecera = (
    "n_Leaver   Re(omega_Leaver)   Im(omega_Leaver)   "
    "residuo   success"
)

np.savetxt(
    "resultados_leaver_l2_60modos.txt",
    datos_l2,
    header=cabecera
)

np.savetxt(
    "resultados_leaver_l3_60modos.txt",
    datos_l3,
    header=cabecera
)

print("\nResultados guardados en unidades de Leaver.")
print("Archivo l=2: resultados_leaver_l2_60modos.txt")
print("Archivo l=3: resultados_leaver_l3_60modos.txt")


""" Gráfica final """

plt.figure(figsize=(8, 6))

plt.plot(
    frec_l2.real,
    -frec_l2.imag,
    "ko-",
    markersize=3,
    linewidth=0.8,
    label=r"$\ell=2$"
)

plt.plot(
    -frec_l2.real,
    -frec_l2.imag,
    "ko-",
    markersize=3,
    linewidth=0.8
)

plt.plot(
    frec_l3.real,
    -frec_l3.imag,
    "k^-",
    markersize=4,
    linewidth=0.8,
    markerfacecolor="none",
    label=r"$\ell=3$"
)

plt.plot(
    -frec_l3.real,
    -frec_l3.imag,
    "k^-",
    markersize=4,
    linewidth=0.8,
    markerfacecolor="none"
)

plt.axvline(0, color="black", linewidth=0.8)

plt.xlabel(r"$\mathrm{Re}(\omega)$")
plt.ylabel(r"$-\mathrm{Im}(\omega)$")

plt.xlim(-1.25, 1.25)
plt.ylim(0, 31)

plt.title(r"Modos quasi-normales de Schwarzschild")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()
