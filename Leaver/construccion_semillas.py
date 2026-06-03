# ============================================================
#  FUNCIÓN AUXILIAR USADA PARA CONSTRUIR LAS SEMILLAS
# ============================================================

def construir_semillas_linspace(n_control, semillas_control):
    """
    Genera guesses intermedios usando np.linspace entre puntos de control.

    Esta función se utilizó durante la fase exploratoria del cálculo para
    construir las listas finales de semillas de los 60 modos. En la ejecución
    final no se llama, porque las semillas refinadas ya están introducidas
    explícitamente en semillas_l2 y semillas_l3.
    """

    n_total = []
    semillas_total = []

    for i in range(len(n_control) - 1):

        n_ini = n_control[i]
        n_fin = n_control[i + 1]

        w_ini = semillas_control[i]
        w_fin = semillas_control[i + 1]

        num_puntos = n_fin - n_ini + 1

        n_segmento = np.arange(n_ini, n_fin + 1)
        semillas_segmento = np.linspace(w_ini, w_fin, num_puntos)

        if i > 0:
            n_segmento = n_segmento[1:]
            semillas_segmento = semillas_segmento[1:]

        n_total.extend(n_segmento)
        semillas_total.extend(semillas_segmento)

    return np.array(n_total), np.array(semillas_total)
