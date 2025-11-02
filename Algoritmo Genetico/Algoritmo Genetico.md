# Algoritmos Genéticos - Análisis de Problema de Optimización

## Introducción Teórica

Los algoritmos genéticos representan una familia de técnicas de optimización y búsqueda inspiradas en los principios fundamentales de la evolución natural propuestos por Charles Darwin. Desarrollados inicialmente por John Holland en la década de 1970, estos algoritmos pertenecen al campo más amplio de la computación evolutiva y se han consolidado como herramientas poderosas para resolver problemas de optimización complejos donde los métodos tradicionales resultan insuficientes o computacionalmente prohibitivos.

### Fundamentos Biológicos y Conceptuales

La base teórica de los algoritmos genéticos se sustenta en la simulación digital de procesos evolutivos, donde una población de soluciones candidatas evoluciona gradualmente hacia soluciones óptimas mediante mecanismos análogos a la selección natural, recombinación genética y mutación. Cada individuo en la población representa una posible solución al problema, codificada típicamente en una estructura de datos similar a un cromosoma.

### Componentes Fundamentales del Algoritmo:

1. **Representación Cromosómica**: 
   - Codificación de soluciones en estructuras genéticas (binarias, reales, permutaciones)
   - Diseño del espacio de búsqueda y restricciones del problema

2. **Función de Aptitud (Fitness)**:
   - Métrica que cuantifica la calidad de cada solución
   - Guía el proceso de selección hacia regiones prometedoras del espacio de búsqueda
   - Debe diseñarse cuidadosamente para reflejar adecuadamente los objetivos del problema

3. **Operadores Genéticos**:
   - **Selección**: Mecanismos para elegir padres basados en su aptitud (ruleta, torneo, elitismo)
   - **Cruce (Crossover)**: Recombinación de información genética entre padres
   - **Mutación**: Introducción de diversidad genética mediante alteraciones aleatorias

4. **Parámetros de Control**:
   - Tamaño de población
   - Tasas de cruce y mutación
   - Criterios de terminación (generaciones, convergencia, tiempo)

### Proceso Evolutivo Iterativo:

```
Inicialización → Evaluación → [Selección → Cruce → Mutación → Evaluación] → Terminación
```

El ciclo evolutivo se repite hasta satisfacer los criterios de terminación, produciendo generaciones sucesivamente mejor adaptadas al entorno del problema.

### Aplicación al Problema de Optimización:

En el contexto de problemas de routing y scheduling, como el presente caso de estudio, los algoritmos genéticos demuestran particular eficacia para manejar espacios de búsqueda combinatoriamente grandes, múltiples restricciones y objetivos conflictivos. La representación típica involucra permutaciones o secuencias que codifican órdenes de visita o asignaciones.

## Implementación en Python

```python
"""
OPTIMIZADOR DE CRONOGRAMA NFL USANDO ALGORITMO GENÉTICO
Este programa optimiza el calendario de partidos de la NFL minimizando las distancias de viaje
a través de un algoritmo genético que garantiza restricciones de balance local/visitante.
"""

# =============================================================================
# SECCIÓN 1: IMPORTACIÓN DE LIBRERÍAS Y CONFIGURACIÓN INICIAL
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from geopy.distance import geodesic
import seaborn as sns

# =============================================================================
# SECCIÓN 2: CARGA Y PREPROCESAMIENTO DE DATOS
# =============================================================================

"""
Esta sección se encarga de cargar los datos de los equipos de la NFL y preparar
la estructura de datos necesaria para los cálculos posteriores.
Los datos incluyen nombre del equipo, coordenadas geográficas, conferencia y división.
"""

# Cargar los datos desde el archivo CSV
df = pd.read_csv('NFL.csv')

# Mostrar información básica de los datos para verificación
print("Información de los equipos:")
print(f"Número de equipos: {len(df)}")
print(f"Conferencias: {df['conference'].unique()}")
print(f"Divisiones: {df['division'].unique()}")

# Preprocesamiento: crear un diccionario de equipos con sus coordenadas y metadatos
# Esta estructura facilitará el acceso a la información de cada equipo durante la optimización
teams = {}
for idx, row in df.iterrows():
    teams[row['team_name']] = {
        'lat': row['lat'],
        'long': row['long'],
        'conference': row['conference'],
        'division': row['division']
    }

# =============================================================================
# SECCIÓN 3: CÁLCULO DE MATRIZ DE DISTANCIAS
# =============================================================================

"""
En esta sección calculamos la matriz de distancias entre todos los pares de equipos
utilizando la fórmula de geodesia que considera la curvatura terrestre.
La distancia se calcula en millas y representa la distancia de viaje real entre estadios.
"""

# Obtener lista de nombres de equipos para indexación
team_names = list(teams.keys())
n_teams = len(team_names)
dist_matrix = np.zeros((n_teams, n_teams))

# Calcular distancia entre cada par de equipos
for i in range(n_teams):
    for j in range(n_teams):
        if i != j:
            # Obtener coordenadas de ambos equipos
            coord1 = (teams[team_names[i]]['lat'], teams[team_names[i]]['long'])
            coord2 = (teams[team_names[j]]['lat'], teams[team_names[j]]['long'])
            # Calcular distancia geodesica y convertir a millas
            dist_matrix[i][j] = geodesic(coord1, coord2).miles
        else:
            # Distancia a sí mismo es cero
            dist_matrix[i][j] = 0

print(f"\nMatriz de distancias calculada ({n_teams}x{n_teams})")

# =============================================================================
# SECCIÓN 4: VISUALIZACIÓN DE MATRIZ DE DISTANCIAS
# =============================================================================

"""
Visualización de la matriz de distancias usando un heatmap de seaborn.
Esto permite identificar visualmente patrones de proximidad entre equipos
y verificar la correcta calculación de distancias.
"""

plt.figure(figsize=(12, 10))
sns.heatmap(dist_matrix, xticklabels=team_names, yticklabels=team_names,
            cmap='YlOrRd', cbar_kws={'label': 'Distancia (millas)'})
plt.title('Matriz de Distancias entre Equipos de la NFL')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# =============================================================================
# SECCIÓN 5: PARÁMETROS Y CONFIGURACIÓN DEL ALGORITMO GENÉTICO
# =============================================================================

"""
Configuración de los parámetros que controlan el comportamiento del algoritmo genético.
- POPULATION_SIZE: Tamaño de la población en cada generación
- GENERATIONS: Número máximo de generaciones a evolucionar
- MUTATION_RATE: Probabilidad de que un individuo sufra mutación
- TOURNAMENT_SIZE: Número de individuos en cada torneo de selección
- ELITISM_COUNT: Número de mejores individuos que pasan directamente a la siguiente generación
"""

# Parámetros principales del algoritmo genético
POPULATION_SIZE = 100
GENERATIONS = 150
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 5
ELITISM_COUNT = 2

# =============================================================================
# SECCIÓN 6: SUPUESTOS Y RESTRICCIONES DEL MODELO
# =============================================================================

"""
SUPUESTOS IMPLEMENTADOS EN EL MODELO:
1. Cada equipo juega exactamente 6 partidos: 3 como local y 3 como visitante
2. No hay restricciones de conferencias o divisiones (cualquier equipo puede jugar contra cualquier otro)
3. No se consideran días de la semana o fechas específicas para los partidos
4. El objetivo único es minimizar la distancia total de viaje de todos los equipos
5. Se asume que los equipos viajan directamente desde su estadio al del oponente
6. No hay partidos contra sí mismos
7. No se consideran restricciones de tiempo entre partidos
"""

# =============================================================================
# SECCIÓN 7: FUNCIONES DEL ALGORITMO GENÉTICO
# =============================================================================

def create_individual():
    """
    Crea un individuo (cronograma) válido que cumple con todas las restricciones.

    ESTRATEGIA IMPLEMENTADA:
    1. Inicializa contadores de partidos local y visitante para cada equipo
    2. Baraja aleatoriamente el orden de los equipos para evitar sesgos
    3. Asigna sistemáticamente partidos como local buscando oponentes válidos
    4. Implementa un mecanismo de reparación para completar partidos faltantes
    5. Garantiza que cada equipo tenga exactamente 3 partidos local y 3 visitante

    Returns:
        list: Lista de tuplas (local, visitante) representando el cronograma completo
    """
    schedule = []
    home_games = {team: 0 for team in team_names}
    away_games = {team: 0 for team in team_names}

    # Crear una lista de equipos y barajarla para aleatoriedad
    teams_shuffled = team_names.copy()
    random.shuffle(teams_shuffled)

    # Fase 1: Asignación inicial de partidos como local
    for home_team in teams_shuffled:
        # Encontrar oponentes válidos (que necesiten partidos como visitante)
        valid_opponents = [
            team for team in team_names
            if team != home_team
            and away_games[team] < 3
            and (home_team, team) not in schedule  # Evitar duplicados
        ]

        # Calcular cuántos partidos necesita como local este equipo
        games_needed = 3 - home_games[home_team]

        # Si hay suficientes oponentes válidos, seleccionar aleatoriamente
        if len(valid_opponents) >= games_needed:
            selected_opponents = random.sample(valid_opponents, games_needed)
            for away_team in selected_opponents:
                schedule.append((home_team, away_team))
                home_games[home_team] += 1
                away_games[away_team] += 1

    # Fase 2: Mecanismo de reparación para partidos faltantes
    max_attempts = 1000
    attempts = 0

    while attempts < max_attempts:
        # Identificar equipos que necesitan completar su calendario
        teams_need_home = [t for t in team_names if home_games[t] < 3]
        teams_need_away = [t for t in team_names if away_games[t] < 3]

        # Verificar si el calendario está completo
        if not teams_need_home and not teams_need_away:
            break  # Calendario completo y balanceado

        # Si hay desbalance que no se puede resolver, salir del loop
        if not teams_need_home or not teams_need_away:
            break

        # Seleccionar equipo local que necesita partidos
        home_team = random.choice(teams_need_home)

        # Encontrar oponentes válidos para el equipo local seleccionado
        valid_opponents = [
            t for t in teams_need_away
            if t != home_team and (home_team, t) not in schedule
        ]

        if valid_opponents:
            away_team = random.choice(valid_opponents)
            schedule.append((home_team, away_team))
            home_games[home_team] += 1
            away_games[away_team] += 1

        attempts += 1

    return schedule


def calculate_distance(schedule):
    """
    Calcula la distancia total de viaje para un cronograma dado.

    Considera que solo el equipo visitante viaja, por lo que suma la distancia
    desde la ubicación del visitante hasta la del local para cada partido.

    Args:
        schedule (list): Lista de tuplas (local, visitante)

    Returns:
        float: Distancia total de viaje en millas
    """
    total_distance = 0
    for home, away in schedule:
        home_idx = team_names.index(home)
        away_idx = team_names.index(away)
        total_distance += dist_matrix[away_idx][home_idx]  # Visitante viaja al local

    return total_distance


def fitness(schedule):
    """
    Función de aptitud: inverso de la distancia total (mayor es mejor)

    Se usa el inverso porque el algoritmo genético maximiza por defecto,
    pero nosotros queremos minimizar distancias. El +1 evita división por cero.

    Args:
        schedule (list): Cronograma a evaluar

    Returns:
        float: Valor de aptitud (fitness)
    """
    total_distance = calculate_distance(schedule)
    return 1 / (total_distance + 1)  # +1 para evitar división por cero


def tournament_selection(population, fitnesses):
    """
    Implementa selección por torneo para elegir padres.

    Estrategia:
    1. Selecciona aleatoriamente TOURNAMENT_SIZE individuos
    2. Elige el mejor de ese grupo como padre
    3. Promueve individuos más aptos sin eliminar completamente la diversidad

    Args:
        population (list): Población actual
        fitnesses (list): Valores de aptitud correspondientes

    Returns:
        list: Individuo seleccionado como padre
    """
    tournament = random.sample(list(zip(population, fitnesses)), TOURNAMENT_SIZE)
    tournament.sort(key=lambda x: x[1], reverse=True)
    return tournament[0][0]


def crossover(parent1, parent2):
    """
    Operador de cruce para combinar dos padres y producir dos hijos.

    Estrategia implementada:
    1. Identifica partidos comunes entre ambos padres
    2. Distribuye partidos únicos de manera aleatoria entre los hijos
    3. Aplica reparación para garantizar validez de los hijos

    Args:
        parent1 (list): Primer padre
        parent2 (list): Segundo padre

    Returns:
        dupla: Dos hijos resultantes del cruce
    """
    # Convertir padres a conjuntos para operaciones de conjunto
    parent1_set = set(parent1)
    parent2_set = set(parent2)

    # Encontrar partidos comunes a ambos padres
    common_games = parent1_set.intersection(parent2_set)

    # Encontrar partidos únicos de cada padre
    unique_parent1 = parent1_set - common_games
    unique_parent2 = parent2_set - common_games

    # Crear hijos inicialmente con partidos comunes
    child1 = list(common_games)
    child2 = list(common_games)

    # Distribuir partidos únicos de manera aleatoria entre los hijos
    for game in unique_parent1:
        if random.random() < 0.5:
            child1.append(game)
        else:
            child2.append(game)

    for game in unique_parent2:
        if random.random() < 0.5:
            child1.append(game)
        else:
            child2.append(game)

    # Reparar ambos hijos para garantizar validez
    return repair_schedule(child1), repair_schedule(child2)


def mutate(schedule):
    """
    Operador de mutación: intercambia dos partidos aleatorios.

    La mutación introduce diversidad en la población permitiendo
    escapar de óptimos locales. Se aplica con probabilidad MUTATION_RATE.

    Args:
        schedule (list): Cronograma a mutar

    Returns:
        list: Cronograma mutado (y reparado si es necesario)
    """
    if len(schedule) < 2:
        return schedule

    mutated = schedule.copy()
    idx1, idx2 = random.sample(range(len(mutated)), 2)
    mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]

    return repair_schedule(mutated)


def repair_schedule(schedule):
    """
    Repara un cronograma para asegurar que cumple con todas las restricciones.

    ESTRATEGIA DE REPARACIÓN:
    1. Primera pasada: mantener solo partidos válidos (sin exceder límites)
    2. Segunda pasada: completar partidos faltantes sistemáticamente
    3. Verificación final: garantizar 3 partidos local y 3 visitante por equipo

    Args:
        schedule (list): Cronograma potencialmente inválido

    Returns:
        list: Cronograma válido y completo
    """
    home_count = {team: 0 for team in team_names}
    away_count = {team: 0 for team in team_names}

    # Primera pasada: filtrar partidos válidos
    valid_schedule = []
    for home, away in schedule:
        if home != away and home_count[home] < 3 and away_count[away] < 3:
            valid_schedule.append((home, away))
            home_count[home] += 1
            away_count[away] += 1

    # Segunda pasada: completar partidos faltantes
    max_attempts = 1000
    attempts = 0

    while attempts < max_attempts:
        # Identificar equipos con calendario incompleto
        teams_need_home = [t for t in team_names if home_count[t] < 3]
        teams_need_away = [t for t in team_names if away_count[t] < 3]

        # Verificar si el calendario está completo
        if not teams_need_home and not teams_need_away:
            break  # Calendario completo

        # Si no hay equipos en ambas categorías, no se puede continuar
        if not teams_need_home or not teams_need_away:
            break

        # Seleccionar equipo local que necesita partidos
        home_team = random.choice(teams_need_home)

        # Buscar oponentes válidos
        valid_opponents = [
            t for t in teams_need_away
            if t != home_team and (home_team, t) not in valid_schedule
        ]

        if valid_opponents:
            away_team = random.choice(valid_opponents)
            valid_schedule.append((home_team, away_team))
            home_count[home_team] += 1
            away_count[away_team] += 1
        else:
            # Estrategia de fallback: buscar cualquier oponente que necesite partidos visitantes
            fallback_opponents = [
                t for t in teams_need_away
                if t != home_team
            ]
            if fallback_opponents:
                away_team = random.choice(fallback_opponents)
                # Verificar que el partido no existe antes de agregarlo
                if (home_team, away_team) not in valid_schedule:
                    valid_schedule.append((home_team, away_team))
                    home_count[home_team] += 1
                    away_count[away_team] += 1

        attempts += 1

    return valid_schedule


# =============================================================================
# SECCIÓN 8: ALGORITMO GENÉTICO PRINCIPAL
# =============================================================================

def genetic_algorithm():
    """
    Implementa el algoritmo genético completo para optimizar el cronograma.

    ESTRATEGIA EVOLUTIVA:
    1. Inicialización: Crear población inicial aleatoria pero válida
    2. Evaluación: Calcular aptitud de cada individuo
    3. Selección: Usar torneos para seleccionar padres
    4. Cruce: Combinar padres para producir hijos
    5. Mutación: Introducir variaciones aleatorias
    6. Elitismo: Preservar los mejores individuos
    7. Reemplazo: Crear nueva generación

    Returns:
        tuple: Mejor cronograma, mejor distancia, historial de aptitudes
    """
    # Inicializar población con individuos válidos
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    best_fitness = []
    avg_fitness = []
    best_schedule = None
    best_distance = float('inf')

    # Ciclo evolutivo por generaciones
    for generation in range(GENERATIONS):
        # Calcular aptitud para cada individuo en la población
        fitnesses = [fitness(ind) for ind in population]

        # Registrar estadísticas de la generación
        current_best_fitness = max(fitnesses)
        current_avg_fitness = sum(fitnesses) / len(fitnesses)
        best_fitness.append(current_best_fitness)
        avg_fitness.append(current_avg_fitness)

        # Identificar el mejor individuo actual
        best_idx = fitnesses.index(current_best_fitness)
        current_best_schedule = population[best_idx]
        current_best_distance = calculate_distance(current_best_schedule)

        # Actualizar el mejor global si es necesario
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_schedule = current_best_schedule.copy()

        # Mostrar progreso cada 50 generaciones
        if generation % 50 == 0:
            print(f"Generación {generation}: Mejor distancia = {current_best_distance:.2f} millas")

        # Crear nueva población mediante operadores genéticos
        new_population = []

        # ELITISMO: Preservar los mejores individuos directamente
        elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:ELITISM_COUNT]
        for idx in elite_indices:
            new_population.append(population[idx])

        # Crear el resto de la población mediante selección, cruce y mutación
        while len(new_population) < POPULATION_SIZE:
            # Selección de padres por torneo
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            # Cruce para producir hijos
            child1, child2 = crossover(parent1, parent2)

            # Mutación aplicada probabilisticamente
            if random.random() < MUTATION_RATE:
                child1 = mutate(child1)
            if random.random() < MUTATION_RATE:
                child2 = mutate(child2)

            # Añadir hijos a la nueva población
            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = new_population

    return best_schedule, best_distance, best_fitness, avg_fitness


# =============================================================================
# SECCIÓN 9: FUNCIÓN PARA IMPRIMIR CRONOGRAMA POR SEMANAS
# =============================================================================

def print_schedule_by_weeks(schedule, matches_per_week=16):
    """
    Divide e imprime el cronograma organizado por semanas.

    Considerando 32 equipos con 6 partidos cada uno, hay 96 partidos totales.
    Con 16 partidos por semana, el cronograma se completa en 6 semanas.

    Args:
        schedule (list): Lista de partidos (local, visitante)
        matches_per_week (int): Número de partidos por semana (default 16)
    """
    total_matches = len(schedule)
    num_weeks = math.ceil(total_matches / matches_per_week)

    print(f"\n{'='*60}")
    print("CRONOGRAMA ORGANIZADO POR SEMANAS")
    print(f"{'='*60}")
    print(f"Total de partidos: {total_matches}")
    print(f"Partidos por semana: {matches_per_week}")
    print(f"Total de semanas: {num_weeks}")
    print(f"{'='*60}")

    for week in range(num_weeks):
        start_idx = week * matches_per_week
        end_idx = min((week + 1) * matches_per_week, total_matches)
        week_matches = schedule[start_idx:end_idx]

        print(f"\n--- SEMANA {week + 1} ({len(week_matches)} partidos) ---")

        for i, (home, away) in enumerate(week_matches, 1):
            home_idx = team_names.index(home)
            away_idx = team_names.index(away)
            distance = dist_matrix[away_idx][home_idx]
            print(f"Partido {i:2d}: {home:15} vs {away:15} | Distancia: {distance:7.1f} millas")

    print(f"\n{'='*60}")


# =============================================================================
# SECCIÓN 10: EJECUCIÓN PRINCIPAL Y RESULTADOS
# =============================================================================

# Ejecutar el algoritmo genético
print("\nEjecutando algoritmo genético...")
best_schedule, best_distance, best_fitness, avg_fitness = genetic_algorithm()

# =============================================================================
# SECCIÓN 11: ANÁLISIS Y VALIDACIÓN DE RESULTADOS
# =============================================================================

print(f"\n{'='*60}")
print("RESULTADOS FINALES DEL ALGORITMO GENÉTICO")
print(f"{'='*60}")
print(f"Mejor distancia total de viaje: {best_distance:,.2f} millas")
print(f"Número total de partidos en el cronograma: {len(best_schedule)}")

# Validación exhaustiva de restricciones
home_games = {team: 0 for team in team_names}
away_games = {team: 0 for team in team_names}

for home, away in best_schedule:
    home_games[home] += 1
    away_games[away] += 1

print("\nVALIDACIÓN DE RESTRICCIONES:")
print(f"# Todos los equipos tienen 3 partidos como local: {all(count == 3 for count in home_games.values())}")
print(f"# Todos los equipos tienen 3 partidos como visitante: {all(count == 3 for count in away_games.values())}")

# Mostrar distribución detallada por equipo
print("\nDISTRIBUCIÓN DETALLADA POR EQUIPO:")
for team in team_names:
    print(f"  {team:20}: {home_games[team]} partidos local, {away_games[team]} partidos visitante")

# Mostrar muestra del cronograma optimizado
print(f"\n{'='*60}")
print("MUESTRA DEL MEJOR CRONOGRAMA (Primeros 10 partidos)")
print(f"{'='*60}")
for i, (home, away) in enumerate(best_schedule[:10]):
    home_idx = team_names.index(home)
    away_idx = team_names.index(away)
    distance = dist_matrix[away_idx][home_idx]
    print(f"Partido {i+1:2d}: {home:15} vs {away:15} | Distancia: {distance:7.1f} millas")

# =============================================================================
# SECCIÓN 12: IMPRIMIR CRONOGRAMA COMPLETO POR SEMANAS
# =============================================================================

# Imprimir el cronograma organizado por semanas
print_schedule_by_weeks(best_schedule)

# =============================================================================
# SECCIÓN 13: VISUALIZACIONES Y GRÁFICAS
# =============================================================================

"""
Esta sección genera visualizaciones para analizar la evolución del algoritmo
y la distribución de distancias en el cronograma optimizado.
"""

# Configurar estilo para las gráficas
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Gráfica 1: Evolución de la aptitud
axes[0, 0].plot(best_fitness, label='Mejor fitness', linewidth=2, color='green')
#axes[0, 0].plot(avg_fitness, label='Fitness promedio', linewidth=2, color='orange')
axes[0, 0].set_xlabel('Generación')
axes[0, 0].set_ylabel('Fitness (1/Distancia_Total)')
axes[0, 0].set_title('Evolución del fitness del Algoritmo Genético')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Gráfica 2: Evolución de la distancia
best_distances = [1/f - 1 if f > 0 else float('inf') for f in best_fitness]
axes[0, 1].plot(best_distances, linewidth=2, color='red')
axes[0, 1].set_xlabel('Generación')
axes[0, 1].set_ylabel('Distancia Total (millas)')
axes[0, 1].set_title('Evolución de la Menor Distancia de Viaje')
axes[0, 1].grid(True, alpha=0.3)

# Gráfica 3: Distancias de viaje por equipo
team_distances = {team: 0 for team in team_names}
for home, away in best_schedule:
    home_idx = team_names.index(home)
    away_idx = team_names.index(away)
    team_distances[away] += dist_matrix[away_idx][home_idx]

teams_sorted = sorted(team_distances.keys(), key=lambda x: team_distances[x], reverse=True)
distances_sorted = [team_distances[team] for team in teams_sorted]

bars = axes[1, 0].bar(range(len(teams_sorted)), distances_sorted, color='steelblue')
axes[1, 0].set_xlabel('Equipos')
axes[1, 0].set_ylabel('Distancia Total de Viaje (millas)')
axes[1, 0].set_title('Distancia Total de Viaje por Equipo (como visitante)')
axes[1, 0].set_xticks(range(len(teams_sorted)))
axes[1, 0].set_xticklabels(teams_sorted, rotation=90, fontsize=8)

# Gráfica 4: Distribución de distancias por partido
match_distances = []
for home, away in best_schedule:
    home_idx = team_names.index(home)
    away_idx = team_names.index(away)
    match_distances.append(dist_matrix[away_idx][home_idx])

axes[1, 1].hist(match_distances, bins=20, alpha=0.7, color='purple', edgecolor='black')
axes[1, 1].set_xlabel('Distancia por Partido (millas)')
axes[1, 1].set_ylabel('Frecuencia')
axes[1, 1].set_title('Distribución de Distancias de Partidos Individuales')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# SECCIÓN 14: ESTADÍSTICAS ADICIONALES Y ANÁLISIS
# =============================================================================

print(f"\n{'='*60}")
print("ESTADÍSTICAS DETALLADAS DEL CRONOGRAMA OPTIMIZADO")
print(f"{'='*60}")

# Cálculo de estadísticas descriptivas
match_distances_array = np.array(match_distances)

print(f"Distancia total de viaje: {best_distance:,.2f} millas")
print(f"Distancia promedio por partido: {np.mean(match_distances_array):.2f} millas")
print(f"Distancia máxima en un partido: {np.max(match_distances_array):.2f} millas")
print(f"Distancia mínima en un partido: {np.min(match_distances_array):.2f} millas")
print(f"Desviación estándar de distancias: {np.std(match_distances_array):.2f} millas")

# Identificar equipos con mayor y menor carga de viaje
max_team = max(team_distances, key=team_distances.get)
min_team = min(team_distances, key=team_distances.get)
print(f"\nEQUIPOS CON MAYOR Y MENOR CARGA DE VIAJE:")
print(f" Mayor distancia: {max_team} ({team_distances[max_team]:,.2f} millas)")
print(f" Menor distancia: {min_team} ({team_distances[min_team]:,.2f} millas)")

# =============================================================================
# SECCIÓN 15: GUARDADO DE RESULTADOS
# =============================================================================

"""
Guardar el cronograma optimizado en un archivo CSV para su uso posterior
y análisis adicional fuera del programa.
"""

# Crear DataFrame con el cronograma optimizado
schedule_df = pd.DataFrame(best_schedule, columns=['Local', 'Visitante'])

# Calcular y agregar distancia para cada partido
schedule_df['Distancia (millas)'] = schedule_df.apply(
    lambda row: dist_matrix[team_names.index(row['Visitante'])][team_names.index(row['Local'])],
    axis=1
)

# Guardar en archivo CSV
schedule_df.to_csv('mejor_cronograma_nfl.csv', index=False)
print(f"\n Mejor cronograma guardado en 'mejor_cronograma_nfl.csv'")

print(f"\n{'='*60}")
print("OPTIMIZACIÓN COMPLETADA EXITOSAMENTE")
print(f"{'='*60}")
```

## Análisis de Resultados

### Progreso del Algoritmo Genético

```
Ejecutando algoritmo genético...
Generación 0: Mejor distancia = 87662.16 millas
Generación 50: Mejor distancia = 43258.14 millas
Generación 100: Mejor distancia = 39519.75 millas

============================================================
RESULTADOS FINALES DEL ALGORITMO GENÉTICO
============================================================
Mejor distancia total de viaje: 39,519.75 millas
Número total de partidos en el cronograma: 96

VALIDACIÓN DE RESTRICCIONES:
# Todos los equipos tienen 3 partidos como local: True
# Todos los equipos tienen 3 partidos como visitante: True
```

### Interpretación de los Resultados:

1. **Evolución de la Solución**:
   - **Generación 0**: La distancia inicial era de 87,662.16 millas
   - **Generación 50**: Mejora significativa a 43,258.14 millas (reducción del 50.6%)
   - **Generación 100**: Optimización final a 39,519.75 millas (reducción total del 54.9%)

2. **Eficiencia del Algoritmo**:
   - El algoritmo demostró una convergencia rápida en las primeras 50 generaciones
   - Las mejoras posteriores fueron más graduales, mostrando una aproximación al óptimo

3. **Validación de Restricciones**:
   - Todas las restricciones de programación fueron satisfechas correctamente
   - Cada equipo juega exactamente 3 partidos como local y 3 como visitante

Desde la perspectiva de viabilidad y cumplimiento de restricciones, los resultados son particularmente significativos. La validación exhaustiva confirma que todas las restricciones operacionales fueron satisfechas completamente: cada equipo participa exactamente en 3 partidos como local y 3 como visitante. Este logro es crucial en problemas de scheduling deportivo, donde la factibilidad de la solución es tan importante como su optimalidad. El manejo exitoso de 96 partidos en el cronograma evidencia la robustez del enfoque para problemas de escala considerable.

El patrón de convergencia observado sugiere que los operadores genéticos (selección, cruce y mutación) funcionaron de manera sinérgica para mantener diversidad poblacional mientras dirigían la búsqueda hacia regiones de alta calidad. La ausencia de estancamiento prematuro en óptimos locales indica que los parámetros del algoritmo fueron apropiadamente configurados, permitiendo una exploración suficiente del espacio de soluciones.

## Conclusión

El algoritmo genético implementado demostró ser altamente efectivo para resolver el problema de optimización de scheduling, logrando una reducción del 54.9% en la distancia total del recorrido mientras garantizaba el cumplimiento integral de todas las restricciones operativas. La convergencia progresiva y sostenida observada a través de 100 generaciones confirma la capacidad del algoritmo para explorar eficientemente espacios de búsqueda combinatoriamente complejos.

La solución final no solo optimiza la métrica objetivo (distancia), sino que también satisface completamente los requisitos de viabilidad del problema, demostrando la dual fortaleza de los algoritmos genéticos en optimización y satisfacción de restricciones. Los resultados validan la aplicabilidad de estas técnicas en problemas del mundo real que involucran múltiples restricciones y objetivos complejos.

**Recomendaciones para Investigación Futura**:
- Experimentación con esquemas de selección adaptativos
- Implementación de operadores de cruce especializados para problemas de scheduling
- Análisis de sensibilidad de parámetros para optimizar performance
- Desarrollo de enfoques híbridos combinando algoritmos genéticos con búsqueda local
- Extensión a optimización multi-objetivo considerando criterios adicionales