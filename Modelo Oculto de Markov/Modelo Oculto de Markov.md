# Modelos Ocultos de Markov (HMM) - Implementación Educativa

## Teoría de Modelos Ocultos de Markov

Los Modelos Ocultos de Markov (HMM) son modelos estadísticos que se utilizan para describir sistemas que pasan por una secuencia de estados no observables (ocultos), pero que producen observaciones visibles. Son una extensión de las cadenas de Markov donde los estados no son directamente observables, pero las salidas (observaciones) sí lo son.

### Componentes de un HMM

Un HMM se define por los siguientes componentes:

1. **Número de estados (N)**: Estados ocultos del sistema
2. **Número de símbolos observables (M)**: Diferentes valores que pueden tomar las observaciones
3. **Matriz de transición (A)**: Probabilidades de transición entre estados
4. **Matriz de emisión (B)**: Probabilidades de emitir cada símbolo desde cada estado
5. **Vector de probabilidades iniciales (π)**: Probabilidad de comenzar en cada estado

### Problemas Fundamentales de los HMM

1. **Problema de evaluación**: Dado un modelo λ = (A, B, π) y una secuencia de observaciones O, calcular P(O|λ)
2. **Problema de decodificación**: Dado λ y O, encontrar la secuencia de estados más probable
3. **Problema de aprendizaje**: Dado O, encontrar los parámetros λ que maximizan P(O|λ)

---

## Fórmulas Matemáticas

### 1. Algoritmo Forward

El algoritmo Forward calcula la probabilidad de observar la secuencia O dado el modelo λ.

**Variable Forward (α)**:

αₜ(i) = P(o₁, o₂, ..., oₜ, qₜ = sᵢ | λ)

**Inicialización** (t=1):

α₁(i) = πᵢ · bᵢ(o₁)  para 1 ≤ i ≤ N

**Recursión** (para t=1,2,...,T-1):

αₜ₊₁(j) = [∑ᵢ₌₁ᴺ αₜ(i) · aᵢⱼ] · bⱼ(oₜ₊₁)  para 1 ≤ j ≤ N

**Terminación**:

P(O|λ) = ∑ᵢ₌₁ᴺ αₜ(i)

### 2. Algoritmo Backward

El algoritmo Backward calcula la probabilidad de observaciones futuras dado el estado actual.

**Variable Backward (β)**:

βₜ(i) = P(oₜ₊₁, oₜ₊₂, ..., oₜ | qₜ = sᵢ, λ)


**Inicialización** (t=T):

βₜ(i) = 1  para 1 ≤ i ≤ N


**Recursión** (para t=T-1, T-2, ..., 1):

βₜ(i) = ∑ⱼ₌₁ᴺ aᵢⱼ · bⱼ(oₜ₊₁) · βₜ₊₁(j)  para 1 ≤ i ≤ N

**Terminación**:

P(O|λ) = ∑ᵢ₌₁ᴺ πᵢ · bᵢ(o₁) · β₁(i)

### 3. Algoritmo de Viterbi

Encuentra la secuencia de estados más probable que genera las observaciones.

**Variable Delta (δ)**:

δₜ(i) = max P(q₁, q₂, ..., qₜ = sᵢ, o₁, o₂, ..., oₜ | λ)
       q₁,q₂,...,qₜ₋₁

**Inicialización** (t=1):

δ₁(i) = πᵢ · bᵢ(o₁)
ψ₁(i) = 0

**Recursión** (para t=2,3,...,T):

δₜ(j) = max [δₜ₋₁(i) · aᵢⱼ] · bⱼ(oₜ)
        1≤i≤N
ψₜ(j) = argmax [δₜ₋₁(i) · aᵢⱼ]
         1≤i≤N

**Terminación**:

P* = max [δₜ(i)]
        1≤i≤N
qₜ* = argmax [δₜ(i)]
           1≤i≤N

**Backtracking** (para t=T-1, T-2, ..., 1):

qₜ* = ψₜ₊₁(qₜ₊₁*)

---

## Problema de Ejemplo: Predicción del Clima

### Descripción del Problema

Imaginemos que queremos predecir el clima (estado oculto) basándonos en las actividades que realizamos (observaciones). No podemos observar directamente el clima, pero podemos observar nuestras actividades, que dependen del clima.

### Estados Ocultos (Clima):
1. **Soleado** (Estado 0)
2. **Nublado** (Estado 1) 
3. **Lluvioso** (Estado 2)

### Observaciones (Actividades):
1. **Caminar** (Observación 0)
2. **Comprar** (Observación 1)
3. **Limpiar** (Observación 2)

### Supuestos del Modelo:

1. **Transiciones del clima**:
   - Si hoy está soleado, mañana probablemente seguirá soleado
   - El clima puede cambiar gradualmente
   
2. **Emisiones (actividades)**:
   - Si está soleado, es más probable que caminemos
   - Si está lluvioso, es más probable que limpiemos en casa

### Matrices del Modelo:

**Matriz de Transición (A)**:
```
          Soleado  Nublado  Lluvioso
Soleado    0.6      0.3       0.1
Nublado    0.4      0.4       0.2
Lluvioso   0.3      0.4       0.3
```

**Matriz de Emisión (B)**:
```
          Caminar  Comprar  Limpiar
Soleado    0.7      0.2       0.1
Nublado    0.4      0.4       0.2
Lluvioso   0.1      0.3       0.6
```

**Probabilidades Iniciales (π)**:
```
Soleado: 0.5, Nublado: 0.3, Lluvioso: 0.2
```

### Preguntas a Responder:

1. **Evaluación**: Dada la secuencia de actividades [Caminar, Comprar, Limpiar, Caminar, Comprar], ¿cuál es la probabilidad de observar esta secuencia?

2. **Decodificación**: ¿Cuál es la secuencia más probable de estados climáticos que generó estas actividades?

---

## Implementación en Python

```python
import numpy as np
from typing import List, Tuple

class ModeloOcultoMarkov:
    """
    Implementación un Modelo Oculto de Markov (HMM).
    
    Un HMM es un modelo estadístico en el que se asume que el sistema
    modelado es un proceso de Markov con estados ocultos (no observables)
    y observaciones visibles que dependen de esos estados.
    """
    
    def __init__(self, n_estados: int, n_simbolos: int):
        """
        Inicializa un Modelo Oculto de Markov.
        
        Parámetros:
        -----------
        n_estados : int
            Número de estados ocultos en el modelo
        n_simbolos : int
            Número de símbolos observables diferentes
        """
        self.n_estados = n_estados
        self.n_simbolos = n_simbolos
        
        # Matriz de transición: A[i][j] = P(estado_j | estado_i)
        # Probabilidad de transitar del estado i al estado j
        self.A = np.random.rand(n_estados, n_estados)
        self.A = self.A / self.A.sum(axis=1, keepdims=True)  # Normalizar
        
        # Matriz de emisión: B[i][k] = P(símbolo_k | estado_i)
        # Probabilidad de observar el símbolo k dado el estado i
        self.B = np.random.rand(n_estados, n_simbolos)
        self.B = self.B / self.B.sum(axis=1, keepdims=True)  # Normalizar
        
        # Probabilidades iniciales: pi[i] = P(estado_i en t=0)
        self.pi = np.random.rand(n_estados)
        self.pi = self.pi / self.pi.sum()  # Normalizar
        
        # Para fines educativos, guardaremos los cálculos intermedios
        self.alpha = None  # Probabilidades forward
        self.beta = None   # Probabilidades backward
        self.delta = None  # Probabilidades de Viterbi
        self.psi = None    # Punteros hacia atrás para Viterbi
    
    def forward(self, observaciones: List[int]) -> float:
        """
        Algoritmo Forward: Calcula la probabilidad de la secuencia observada.
        
        El algoritmo forward calcula la probabilidad de observar la secuencia
        dada, sumando sobre todas las posibles secuencias de estados ocultos.
        
        Parámetros:
        -----------
        observaciones : List[int]
            Secuencia de observaciones (índices de símbolos)
            
        Retorna:
        --------
        float : Probabilidad de la secuencia observada
        """
        T = len(observaciones)
        self.alpha = np.zeros((T, self.n_estados))
        
        # Paso 1: Inicialización (t=0)
        for i in range(self.n_estados):
            self.alpha[0][i] = self.pi[i] * self.B[i][observaciones[0]]
        
        # Paso 2: Inducción (para t=1 hasta T-1)
        for t in range(1, T):
            for j in range(self.n_estados):
                suma = 0
                for i in range(self.n_estados):
                    suma += self.alpha[t-1][i] * self.A[i][j]
                self.alpha[t][j] = suma * self.B[j][observaciones[t]]
        
        # Paso 3: Terminación
        probabilidad = sum(self.alpha[T-1])
        return probabilidad
    
    def backward(self, observaciones: List[int]) -> float:
        """
        Algoritmo Backward: Calcula la probabilidad de la secuencia observada.
        
        Similar al algoritmo forward, pero calcula desde el final hacia el inicio.
        
        Parámetros:
        -----------
        observaciones : List[int]
            Secuencia de observaciones (índices de símbolos)
            
        Retorna:
        --------
        float : Probabilidad de la secuencia observada
        """
        T = len(observaciones)
        self.beta = np.zeros((T, self.n_estados))
        
        # Paso 1: Inicialización (t=T-1)
        for i in range(self.n_estados):
            self.beta[T-1][i] = 1.0  # Por definición
            
        # Paso 2: Inducción (para t=T-2 hasta 0)
        for t in range(T-2, -1, -1):
            for i in range(self.n_estados):
                suma = 0
                for j in range(self.n_estados):
                    suma += self.A[i][j] * self.B[j][observaciones[t+1]] * self.beta[t+1][j]
                self.beta[t][i] = suma
        
        # Paso 3: Terminación
        probabilidad = 0
        for i in range(self.n_estados):
            probabilidad += self.pi[i] * self.B[i][observaciones[0]] * self.beta[0][i]
            
        return probabilidad
    
    def viterbi(self, observaciones: List[int]) -> Tuple[List[int], float]:
        """
        Algoritmo de Viterbi: Encuentra la secuencia de estados más probable.
        
        Este algoritmo encuentra la secuencia de estados ocultos que maximiza
        la probabilidad de observar la secuencia dada.
        
        Parámetros:
        -----------
        observaciones : List[int]
            Secuencia de observaciones (índices de símbolos)
            
        Retorna:
        --------
        Tuple[List[int], float] : 
            - Secuencia de estados más probable
            - Probabilidad de esta secuencia
        """
        T = len(observaciones)
        self.delta = np.zeros((T, self.n_estados))
        self.psi = np.zeros((T, self.n_estados), dtype=int)
        
        # Paso 1: Inicialización (t=0)
        for i in range(self.n_estados):
            self.delta[0][i] = self.pi[i] * self.B[i][observaciones[0]]
            self.psi[0][i] = 0
        
        # Paso 2: Recursión (para t=1 hasta T-1)
        for t in range(1, T):
            for j in range(self.n_estados):
                max_valor = 0
                max_estado = 0
                for i in range(self.n_estados):
                    valor = self.delta[t-1][i] * self.A[i][j]
                    if valor > max_valor:
                        max_valor = valor
                        max_estado = i
                
                self.delta[t][j] = max_valor * self.B[j][observaciones[t]]
                self.psi[t][j] = max_estado
        
        # Paso 3: Terminación
        max_prob = np.max(self.delta[T-1])
        mejor_estado_final = np.argmax(self.delta[T-1])
        
        # Paso 4: Reconstrucción de la secuencia de estados (backtracking)
        mejor_secuencia = [mejor_estado_final]
        
        for t in range(T-1, 0, -1):
            mejor_estado_final = self.psi[t][mejor_estado_final]
            mejor_secuencia.insert(0, mejor_estado_final)
            
        return mejor_secuencia, max_prob
    
    def verificar_consistencia(self, observaciones: List[int]) -> bool:
        """
        Verifica que forward y backward den el mismo resultado.
        
        Parámetros:
        -----------
        observaciones : List[int]
            Secuencia de observaciones
            
        Retorna:
        --------
        bool : True si los resultados son consistentes
        """
        p_forward = self.forward(observaciones)
        p_backward = self.backward(observaciones)
        
        # Debido a posibles errores de redondeo en punto flotante,
        # comparamos con una tolerancia pequeña
        return abs(p_forward - p_backward) < 1e-10
    
    def mostrar_parametros(self):
        """Muestra los parámetros del modelo de forma legible."""
        print("=" * 60)
        print("PARÁMETROS DEL MODELO OCULTO DE MARKOV")
        print("=" * 60)
        print(f"Número de estados: {self.n_estados}")
        print(f"Número de símbolos observables: {self.n_simbolos}")
        print("\nMatriz de transición (A):")
        for i in range(self.n_estados):
            print(f"  Estado {i}: {self.A[i]}")
        
        print("\nMatriz de emisión (B):")
        for i in range(self.n_estados):
            print(f"  Estado {i}: {self.B[i]}")
        
        print(f"\nProbabilidades iniciales (π): {self.pi}")
        print("=" * 60)


# ============================================================================
# EJEMPLO DE USO: PREDICCIÓN DEL CLIMA
# ============================================================================

def ejemplo_clima():
    """
    Ejemplo: Modelo del clima con estados ocultos.
    
    Estados ocultos (no observables directamente):
      0: Soleado
      1: Nublado
      2: Lluvioso
    
    Observaciones (actividades que realizamos):
      0: Caminar
      1: Comprar
      2: Limpiar
    """
    print("\n" + "="*60)
    print("EJEMPLO: MODELO DE PREDICCIÓN DEL CLIMA")
    print("="*60)
    
    # Crear el modelo
    modelo = ModeloOcultoMarkov(n_estados=3, n_simbolos=3)
    
    # Definir parámetros del modelo (en lugar de usar valores aleatorios)
    # Para fines educativos, usaremos probabilidades intuitivas
    
    # Matriz de transición del clima
    # Soleado -> Soleado: 0.6, Soleado -> Nublado: 0.3, Soleado -> Lluvioso: 0.1
    # Nublado -> Soleado: 0.4, Nublado -> Nublado: 0.4, Nublado -> Lluvioso: 0.2
    # Lluvioso -> Soleado: 0.3, Lluvioso -> Nublado: 0.4, Lluvioso -> Lluvioso: 0.3
    modelo.A = np.array([
        [0.6, 0.3, 0.1],  # Transiciones desde Soleado
        [0.4, 0.4, 0.2],  # Transiciones desde Nublado
        [0.3, 0.4, 0.3]   # Transiciones desde Lluvioso
    ])
    
    # Matriz de emisión (qué actividad hacemos dado el clima)
    # Si está soleado: Caminar (0.7), Comprar (0.2), Limpiar (0.1)
    # Si está nublado: Caminar (0.4), Comprar (0.4), Limpiar (0.2)
    # Si está lluvioso: Caminar (0.1), Comprar (0.3), Limpiar (0.6)
    modelo.B = np.array([
        [0.7, 0.2, 0.1],  # Emisiones desde Soleado
        [0.4, 0.4, 0.2],  # Emisiones desde Nublado
        [0.1, 0.3, 0.6]   # Emisiones desde Lluvioso
    ])
    
    # Probabilidades iniciales del clima
    # Soleado: 0.5, Nublado: 0.3, Lluvioso: 0.2
    modelo.pi = np.array([0.5, 0.3, 0.2])
    
    # Mostrar parámetros del modelo
    modelo.mostrar_parametros()
    
    # Secuencia de observaciones (actividades durante 5 días)
    # 0: Caminar, 1: Comprar, 2: Limpiar
    observaciones = [0, 1, 2, 0, 1]  # Ejemplo: Caminar, Comprar, Limpiar, Caminar, Comprar
    
    print(f"\nSecuencia de observaciones: {observaciones}")
    print("  (0=Caminar, 1=Comprar, 2=Limpiar)")
    
    # 1. Calcular probabilidad usando Forward
    prob_forward = modelo.forward(observaciones)
    print(f"\n1. Probabilidad (Forward): {prob_forward:.6f}")
    
    # 2. Calcular probabilidad usando Backward
    prob_backward = modelo.backward(observaciones)
    print(f"2. Probabilidad (Backward): {prob_backward:.6f}")
    
    # 3. Verificar consistencia entre Forward y Backward
    consistente = modelo.verificar_consistencia(observaciones)
    print(f"3. Resultados consistentes: {consistente}")
    
    # 4. Encontrar secuencia más probable de estados (clima) usando Viterbi
    mejor_secuencia, mejor_prob = modelo.viterbi(observaciones)
    
    print(f"\n4. Secuencia más probable de estados (Viterbi):")
    print(f"   Probabilidad: {mejor_prob:.6f}")
    print(f"   Secuencia: {mejor_secuencia}")
    
    # Traducir estados a nombres
    nombres_estados = ["Soleado", "Nublado", "Lluvioso"]
    secuencia_nombres = [nombres_estados[estado] for estado in mejor_secuencia]
    print(f"   Interpretación: {secuencia_nombres}")
    
    # 5. Mostrar cálculos intermedios (para fines educativos)
    print("\n5. Cálculos intermedios (para t=0, t=1, t=2):")
    print(f"   Alpha (Forward):")
    for t in range(min(3, len(observaciones))):
        print(f"     t={t}: {modelo.alpha[t]}")
    
    print(f"\n   Delta (Viterbi):")
    for t in range(min(3, len(observaciones))):
        print(f"     t={t}: {modelo.delta[t]}")
    
    print("\n" + "="*60)
    print("INTERPRETACIÓN DE RESULTADOS:")
    print("="*60)
    print("1. La probabilidad de observar la secuencia de actividades")
    print(f"   es aproximadamente {prob_forward:.6f}.")
    print("\n2. La secuencia más probable de estados climáticos es:")
    for dia, estado in enumerate(secuencia_nombres):
        print(f"   Día {dia+1}: {estado}")
    print("\n3. Esto significa que dados los datos observados (actividades),")
    print("   esta es la secuencia de clima más probable que las generó.")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal que ejecuta el ejemplo."""
    print("IMPLEMENTACIÓN DE MODELOS OCULTOS DE MARKOV")
    print("="*60)
    print("Este programa implementa:")
    print("1. Algoritmo Forward: Calcula la probabilidad de una secuencia")
    print("2. Algoritmo Backward: Versión alternativa de Forward")
    print("3. Algoritmo de Viterbi: Encuentra la secuencia de estados más probable")
    print("\nLos algoritmos se aplicarán a un ejemplo de predicción del clima.")
    
    # Ejecutar el ejemplo
    ejemplo_clima()
    


if __name__ == "__main__":
    main()

```

---

## Análisis de Resultados

### Resultados Esperados del Ejemplo:

Para la secuencia de observaciones [0, 1, 2, 0, 1] (Caminar, Comprar, Limpiar, Caminar, Comprar):

1. **Probabilidad de la secuencia** (Forward/Backward):
   - Debería ser un valor entre 0 y 1
   - Forward y Backward deben dar el mismo resultado (consistencia)

2. **Secuencia más probable de estados** (Viterbi):
   - Una secuencia de 5 estados (uno por cada día)
   - Cada estado será 0 (Soleado), 1 (Nublado) o 2 (Lluvioso)

3. **Interpretación**:
   - Podemos analizar cómo las actividades observadas se correlacionan con el clima inferido
   - Ejemplo: Si observamos "Limpiar" (2), es más probable que el estado sea "Lluvioso" (2)

### Análisis de las Matrices de Probabilidad:

- **Matriz α (Forward)**: Muestra cómo evoluciona la probabilidad de estar en cada estado a lo largo del tiempo
- **Matriz δ (Viterbi)**: Muestra la máxima probabilidad acumulada para llegar a cada estado
- **Matriz ψ (Punteros)**: Permite reconstruir la ruta óptima

---

## Conclusiones

### 1. Aprendizajes Clave

Los Modelos Ocultos de Markov proporcionan un marco matemático robusto para:
- **Modelar sistemas con estados no observables**
- **Inferir secuencias de estados ocultos a partir de observaciones**
- **Calcular probabilidades de secuencias observadas**

### 2. Aplicaciones Prácticas

Los HMM tienen numerosas aplicaciones en el mundo real:
- **Reconocimiento de voz**: Los estados representan fonemas, las observaciones son señales acústicas
- **Bioinformática**: Análisis de secuencias de ADN y proteínas
- **Finanzas**: Modelado de series temporales y detección de regímenes de mercado
- **Procesamiento de lenguaje natural**: Etiquetado gramatical (POS tagging)

### 3. Ventajas del Enfoque

- **Eficiencia computacional**: Los algoritmos Forward, Backward y Viterbi tienen complejidad O(N²T)
- **Fundamento teórico sólido**: Basado en teoría de probabilidad y procesos estocásticos
- **Interpretabilidad**: Los resultados pueden interpretarse en términos de probabilidades

### 4. Limitaciones

- **Suposición de Markov**: El estado futuro solo depende del estado actual
- **Estacionariedad**: Los parámetros A, B, π no cambian con el tiempo
- **Independencia de observaciones**: Dado el estado actual, la observación es independiente de todo lo demás

### 5. Extensión del Ejemplo

Este ejemplo educativo puede extenderse para:
- **Aprendizaje de parámetros**: Implementar el algoritmo de Baum-Welch (Expectation-Maximization)
- **Modelos más complejos**: Aumentar el número de estados o observaciones
- **Aplicaciones específicas**: Adaptar el modelo a problemas reales particulares

### 6. Reflexión Final

La implementación presentada demuestra cómo las matemáticas abstractas (probabilidades, matrices, recursión) se traducen en algoritmos concretos que resuelven problemas prácticos. La elegancia de los algoritmos Forward, Backward y Viterbi radica en cómo descomponen un problema exponencialmente complejo (considerar todas las posibles secuencias de estados) en un problema polinomial mediante programación dinámica.

Esta implementación sirve como base para entender conceptos más avanzados en aprendizaje automático y procesamiento de señales, donde los modelos probabilísticos juegan un papel fundamental.
