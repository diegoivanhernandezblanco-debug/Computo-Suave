import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==========================================================
# CONFIGURACIÓN GENERAL
# ==========================================================

NUM_CIUDADES = 25
TAM_POBLACION = 180
GENERACIONES = 500
PROB_MUTACION = 0.18
TAM_TORNEO = 5
ELITISMO = 8
SEMILLA = 25

np.random.seed(SEMILLA)

# Cada ciudad tiene una coordenada X y Y
ciudades = np.random.randint(5, 96, size=(NUM_CIUDADES, 2))

# Se agregan nombres para que el resultado sea más entendible
nombres = [f"C{i + 1}" for i in range(NUM_CIUDADES)]


# ==========================================================
# FUNCIONES DEL ALGORITMO GENÉTICO
# ==========================================================

def calcular_distancia(ruta):
    """
    Calcula la distancia total recorrida.
    Al terminar, regresa de la última ciudad a la primera.
    """
    recorrido = ciudades[ruta]
    recorrido_cerrado = np.vstack([recorrido, recorrido[0]])

    diferencias = np.diff(recorrido_cerrado, axis=0)
    distancias = np.sqrt(np.sum(diferencias ** 2, axis=1))

    return np.sum(distancias)


def crear_poblacion():
    """
    Crea rutas aleatorias.
    Cada individuo representa un posible recorrido.
    """
    return np.array([
        np.random.permutation(NUM_CIUDADES)
        for _ in range(TAM_POBLACION)
    ])


def evaluar_poblacion(poblacion):
    """
    Evalúa las rutas.
    Una distancia menor representa una mejor solución.
    """
    return np.array([
        calcular_distancia(individuo)
        for individuo in poblacion
    ])


def seleccion_por_torneo(poblacion, distancias):
    """
    Selecciona varios individuos aleatorios y devuelve
    el que tenga la menor distancia.
    """
    participantes = np.random.choice(
        len(poblacion),
        size=TAM_TORNEO,
        replace=False
    )

    ganador = participantes[
        np.argmin(distancias[participantes])
    ]

    return poblacion[ganador].copy()


def cruce_ordenado(padre1, padre2):
    """
    Ordered Crossover (OX).
    Conserva una parte del primer padre y completa la ruta
    usando el orden de las ciudades del segundo padre.
    """
    inicio, fin = sorted(
        np.random.choice(NUM_CIUDADES, size=2, replace=False)
    )

    hijo = np.full(NUM_CIUDADES, -1, dtype=int)

    # Copiar el segmento del primer padre
    hijo[inicio:fin + 1] = padre1[inicio:fin + 1]

    # Ciudades del segundo padre que todavía no están en el hijo
    ciudades_faltantes = [
        ciudad for ciudad in padre2
        if ciudad not in hijo
    ]

    posiciones_vacias = np.where(hijo == -1)[0]

    for posicion, ciudad in zip(posiciones_vacias, ciudades_faltantes):
        hijo[posicion] = ciudad

    return hijo


def mutar(ruta):
    """
    Mutación por inversión.
    Invierte una sección de la ruta con cierta probabilidad.
    """
    if np.random.random() < PROB_MUTACION:
        inicio, fin = sorted(
            np.random.choice(NUM_CIUDADES, size=2, replace=False)
        )

        ruta[inicio:fin + 1] = ruta[inicio:fin + 1][::-1]

    return ruta


def crear_nueva_generacion(poblacion, distancias):
    """
    Conserva a los mejores individuos y crea nuevos hijos.
    """
    orden = np.argsort(distancias)

    nueva_poblacion = [
        poblacion[i].copy()
        for i in orden[:ELITISMO]
    ]

    while len(nueva_poblacion) < TAM_POBLACION:
        padre1 = seleccion_por_torneo(poblacion, distancias)
        padre2 = seleccion_por_torneo(poblacion, distancias)

        hijo = cruce_ordenado(padre1, padre2)
        hijo = mutar(hijo)

        nueva_poblacion.append(hijo)

    return np.array(nueva_poblacion)


# ==========================================================
# EJECUCIÓN DEL ALGORITMO
# ==========================================================

print("=" * 62)
print("   ALGORITMO GENÉTICO PARA OPTIMIZACIÓN DE RUTAS")
print("=" * 62)
print(f"Ciudades: {NUM_CIUDADES}")
print(f"Población: {TAM_POBLACION}")
print(f"Generaciones: {GENERACIONES}")
print(f"Probabilidad de mutación: {PROB_MUTACION * 100:.0f}%")
print("-" * 62)

poblacion = crear_poblacion()

mejor_ruta_global = None
mejor_distancia_global = float("inf")

historial_mejor = []
historial_promedio = []
rutas_guardadas = []

for generacion in range(GENERACIONES):
    distancias = evaluar_poblacion(poblacion)

    indice_mejor = np.argmin(distancias)
    mejor_distancia_actual = distancias[indice_mejor]
    mejor_ruta_actual = poblacion[indice_mejor].copy()

    if mejor_distancia_actual < mejor_distancia_global:
        mejor_distancia_global = mejor_distancia_actual
        mejor_ruta_global = mejor_ruta_actual.copy()

    historial_mejor.append(mejor_distancia_global)
    historial_promedio.append(np.mean(distancias))

    # Guardar algunas rutas para la animación
    if generacion % 5 == 0:
        rutas_guardadas.append(
            (generacion, mejor_ruta_global.copy(), mejor_distancia_global)
        )

    if generacion == 0 or (generacion + 1) % 50 == 0:
        print(
            f"Generación {generacion + 1:3d} | "
            f"Mejor distancia: {mejor_distancia_global:8.2f} | "
            f"Promedio: {np.mean(distancias):8.2f}"
        )

    poblacion = crear_nueva_generacion(poblacion, distancias)


# ==========================================================
# RESULTADO FINAL
# ==========================================================

ruta_nombres = [nombres[i] for i in mejor_ruta_global]
ruta_nombres.append(ruta_nombres[0])

print("-" * 62)
print("PROCESO TERMINADO")
print(f"Mejor distancia encontrada: {mejor_distancia_global:.2f}")
print("\nMejor recorrido:")

for i in range(0, len(ruta_nombres), 8):
    print(" -> ".join(ruta_nombres[i:i + 8]))

print("=" * 62)


# ==========================================================
# GRÁFICA PRINCIPAL
# ==========================================================

fig, (ax_ruta, ax_evolucion) = plt.subplots(
    1,
    2,
    figsize=(15, 7)
)

fig.suptitle(
    "Optimización de rutas mediante un algoritmo genético",
    fontsize=16,
    fontweight="bold"
)

# ----- Gráfica de la mejor ruta -----

coordenadas_ruta = ciudades[mejor_ruta_global]
coordenadas_ruta = np.vstack([
    coordenadas_ruta,
    coordenadas_ruta[0]
])

ax_ruta.plot(
    coordenadas_ruta[:, 0],
    coordenadas_ruta[:, 1],
    color="#00B4D8",
    linewidth=2.2,
    marker="o",
    markersize=7
)

ax_ruta.scatter(
    ciudades[:, 0],
    ciudades[:, 1],
    color="#EF233C",
    s=65,
    zorder=3,
    edgecolors="black"
)

for i, (x, y) in enumerate(ciudades):
    ax_ruta.annotate(
        nombres[i],
        (x, y),
        xytext=(5, 6),
        textcoords="offset points",
        fontsize=8,
        fontweight="bold"
    )

# Resaltar la ciudad inicial
inicio = mejor_ruta_global[0]

ax_ruta.scatter(
    ciudades[inicio, 0],
    ciudades[inicio, 1],
    color="#70E000",
    s=170,
    marker="*",
    edgecolors="black",
    zorder=4,
    label="Inicio y final"
)

ax_ruta.set_title(
    f"Mejor ruta encontrada\nDistancia total: "
    f"{mejor_distancia_global:.2f}"
)

ax_ruta.set_xlabel("Coordenada X")
ax_ruta.set_ylabel("Coordenada Y")
ax_ruta.grid(alpha=0.25)
ax_ruta.legend()


# ----- Gráfica de evolución -----

ax_evolucion.plot(
    historial_promedio,
    color="#FF9F1C",
    alpha=0.65,
    label="Promedio de la población"
)

ax_evolucion.plot(
    historial_mejor,
    color="#2EC4B6",
    linewidth=2.5,
    label="Mejor solución"
)

ax_evolucion.fill_between(
    range(GENERACIONES),
    historial_mejor,
    historial_promedio,
    color="#2EC4B6",
    alpha=0.10
)

ax_evolucion.set_title("Evolución del algoritmo")
ax_evolucion.set_xlabel("Generación")
ax_evolucion.set_ylabel("Distancia")
ax_evolucion.grid(alpha=0.25)
ax_evolucion.legend()

plt.tight_layout()

# Guardar automáticamente la evidencia
plt.savefig(
    "resultado_algoritmo_genetico.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# ANIMACIÓN OPCIONAL
# ==========================================================

respuesta = input(
    "\n¿Deseas visualizar la evolución animada? (s/n): "
).strip().lower()

if respuesta == "s":
    fig_animacion, ax_animacion = plt.subplots(figsize=(9, 7))

    def actualizar_animacion(frame):
        ax_animacion.clear()

        generacion, ruta, distancia = rutas_guardadas[frame]

        puntos = ciudades[ruta]
        puntos = np.vstack([puntos, puntos[0]])

        ax_animacion.plot(
            puntos[:, 0],
            puntos[:, 1],
            color="#00B4D8",
            linewidth=2,
            marker="o"
        )

        ax_animacion.scatter(
            ciudades[:, 0],
            ciudades[:, 1],
            color="#EF233C",
            s=60,
            edgecolors="black",
            zorder=3
        )

        for i, (x, y) in enumerate(ciudades):
            ax_animacion.annotate(
                nombres[i],
                (x, y),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8
            )

        ax_animacion.set_title(
            f"Generación {generacion + 1}\n"
            f"Mejor distancia: {distancia:.2f}"
        )

        ax_animacion.set_xlabel("Coordenada X")
        ax_animacion.set_ylabel("Coordenada Y")
        ax_animacion.grid(alpha=0.25)
        ax_animacion.set_xlim(0, 100)
        ax_animacion.set_ylim(0, 100)

    animacion = FuncAnimation(
        fig_animacion,
        actualizar_animacion,
        frames=len(rutas_guardadas),
        interval=100,
        repeat=False
    )

    plt.show()