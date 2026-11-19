import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Variable de entrada: Temperatura
temperatura = ctrl.Antecedent(np.arange(15, 41, 1), 'temperatura')

# Variable de salida: Velocidad del minisplit
velocidad = ctrl.Consequent(np.arange(0, 101, 1), 'velocidad')

# Conjuntos difusos de temperatura
temperatura['frio'] = fuzz.trimf(temperatura.universe, [15, 15, 22])
temperatura['templado'] = fuzz.trimf(temperatura.universe, [20, 25, 30])
temperatura['caliente'] = fuzz.trimf(temperatura.universe, [28, 40, 40])

# Conjuntos difusos de velocidad
velocidad['baja'] = fuzz.trimf(velocidad.universe, [0, 0, 40])
velocidad['media'] = fuzz.trimf(velocidad.universe, [30, 50, 70])
velocidad['alta'] = fuzz.trimf(velocidad.universe, [60, 100, 100])

# Reglas difusas
regla1 = ctrl.Rule(temperatura['frio'], velocidad['baja'])
regla2 = ctrl.Rule(temperatura['templado'], velocidad['media'])
regla3 = ctrl.Rule(temperatura['caliente'], velocidad['alta'])

# Sistema de control
control_aire = ctrl.ControlSystem([regla1, regla2, regla3])
simulacion = ctrl.ControlSystemSimulation(control_aire)

# Entrada del usuario
temp_actual = float(input("Ingrese la temperatura de la habitación: "))
simulacion.input['temperatura'] = temp_actual

# Procesar
simulacion.compute()

# Resultado
vel = simulacion.output['velocidad']

print(f"\nTemperatura actual: {temp_actual} °C")
print(f"Velocidad recomendada: {vel:.2f}%")

if temp_actual >= 30:
    print("Acción: Habitación muy caliente -> Ajustar minisplit a 18°C")
elif temp_actual >= 25:
    print("Acción: Mantener temperatura entre 20°C y 22°C")
else:
    print("Acción: Mantener velocidad baja")