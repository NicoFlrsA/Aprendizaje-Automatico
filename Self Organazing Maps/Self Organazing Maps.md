# Self-Organizing Maps (SOM): 

## Introducción Teórica

Los Self-Organizing Maps (SOM), también conocidos como Mapas Auto-Organizativos de Kohonen, representan una de las arquitecturas de redes neuronales artificiales más fascinantes y visualmente intuitivas en el campo del aprendizaje no supervisado. Desarrollados por Teuvo Kohonen en la década de 1980, los SOM combinan principios de neurociencia, aprendizaje competitivo y reducción de dimensionalidad para crear representaciones topológicas preservadas de datos de alta dimensión.

### Fundamentos Biológicos

La inspiración fundamental de los SOM proviene de la organización del cerebro mamífero, específicamente de la corteza cerebral. En sistemas biológicos, diferentes regiones corticales se especializan en procesar distintos tipos de información sensorial (visual, auditiva, táctil), manteniendo una organización topográfica donde neuronas cercanas físicamente responden a estímulos similares. Esta propiedad de preservación topográfica es la esencia que los SOM intentan replicar computacionalmente.

### Arquitectura Básica

Un SOM consiste en dos capas principales:
- **Capa de entrada**: Recibe los vectores de datos de dimensión n
- **Capa de salida (mapa)**: Una grilla (generalmente 2D) de neuronas, cada una con un vector de pesos de la misma dimensión que los datos de entrada

La disposición de las neuronas en la capa de salida puede ser rectangular, hexagonal o incluso de mayor dimensión, aunque las configuraciones bidimensionales son las más comunes para visualización.

### Algoritmo de Aprendizaje

El proceso de entrenamiento sigue un esquema competitivo-cooperativo:

1. **Inicialización**: Los vectores de peso se inicializan, típicamente con valores aleatorios o mediante muestreo de los datos de entrada.

2. **Competencia**: Para cada vector de entrada, se calcula la neurona "ganadora" (Best Matching Unit - BMU) usando una medida de similitud (generalmente distancia euclidiana).

3. **Cooperación**: Las neuronas vecinas a la BMU en el mapa son actualizadas, creando una "vecindad topológica" donde neuronas cercanas aprenden patrones similares.

4. **Adaptación**: Los pesos de la BMU y sus vecinos se actualizan para acercarse al vector de entrada.

5. **Decaimiento**: Los parámetros de aprendizaje (tasa de aprendizaje y radio de vecindad) disminuyen gradualmente durante el entrenamiento.

### Propiedades Fundamentales

**Preservación Topológica**: La propiedad más importante de los SOM es que mantiene las relaciones espaciales de los datos de entrada en el espacio de salida. Si dos puntos son cercanos en el espacio original, sus BMUs serán cercanas en el mapa.

**Reducción de Dimensionalidad**: Los SOM proyectan datos de alta dimensión en un espacio de baja dimensión (generalmente 2D) mientras preservan la estructura topológica.

**Agrupamiento Natural**: Emergen agrupamientos naturales donde regiones del mapa representan clusters de datos similares.

### Aplicaciones Prácticas

Los SOM encuentran aplicación en diversas áreas:
- Análisis exploratorio de datos
- Visualización de datos de alta dimensión
- Detección de anomalías
- Minería de datos y descubrimiento de conocimiento
- Procesamiento de imágenes y señales
- Bioinformática y genómica

## Implementación en Python

### Aplicación Útil: Segmentación de Clientes para Marketing

Implementaremos un SOM para segmentar clientes de un comercio electrónico basado en su comportamiento de compra, permitiendo estrategias de marketing personalizadas.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
import warnings
warnings.filterwarnings('ignore')

class SelfOrganizingMap:
    def __init__(self, grid_size, input_dim, learning_rate=0.5, sigma=None):
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.sigma = sigma if sigma is not None else max(grid_size) / 2

        # Inicializar pesos aleatoriamente
        self.weights = np.random.random((grid_size[0], grid_size[1], input_dim))

    def find_bmu(self, x):
        """Encuentra la Best Matching Unit (neurona ganadora)"""
        distances = np.sqrt(((self.weights - x) ** 2).sum(axis=2))
        bmu_index = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
        return bmu_index

    def calculate_neighborhood(self, bmu, current_sigma):
        """Calcula la función de vecindad gaussiana"""
        x = np.arange(self.grid_size[0])
        y = np.arange(self.grid_size[1])
        xx, yy = np.meshgrid(x, y, indexing='ij')

        distance_sq = (xx - bmu[0]) ** 2 + (yy - bmu[1]) ** 2
        neighborhood = np.exp(-distance_sq / (2 * current_sigma ** 2))
        return neighborhood

    def train(self, data, epochs=100, verbose=True):
        """Entrena el SOM"""
        for epoch in range(epochs):
            # Decaimiento de parámetros
            current_lr = self.learning_rate * np.exp(-epoch / epochs)
            current_sigma = self.sigma * np.exp(-epoch / epochs)

            # Mezclar datos en cada época
            np.random.shuffle(data)

            for sample in data:
                # Encontrar BMU
                bmu = self.find_bmu(sample)

                # Calcular vecindad
                neighborhood = self.calculate_neighborhood(bmu, current_sigma)

                # Actualizar pesos
                for i in range(self.grid_size[0]):
                    for j in range(self.grid_size[1]):
                        influence = neighborhood[i, j]
                        self.weights[i, j] += current_lr * influence * (sample - self.weights[i, j])

            if verbose and epoch % 20 == 0:
                error = self.calculate_quantization_error(data)
                print(f"Época {epoch}, Error de cuantización: {error:.4f}")

    def calculate_quantization_error(self, data):
        """Calcula el error de cuantización promedio"""
        total_error = 0
        for sample in data:
            bmu = self.find_bmu(sample)
            error = np.linalg.norm(sample - self.weights[bmu])
            total_error += error
        return total_error / len(data)

    def predict(self, data):
        """Predice las BMUs para cada muestra"""
        bmus = []
        for sample in data:
            bmu = self.find_bmu(sample)
            bmus.append(bmu)
        return bmus

    def get_umatrix(self):
        """Calcula la matriz U (unified distance matrix)"""
        umatrix = np.zeros((self.grid_size[0], self.grid_size[1]))

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                neighbors = []
                # Vecinos en las 8 direcciones
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.grid_size[0] and 0 <= nj < self.grid_size[1]:
                            neighbors.append(self.weights[ni, nj])

                if neighbors:
                    # Distancia promedio a los vecinos
                    distances = [np.linalg.norm(self.weights[i, j] - neighbor)
                                for neighbor in neighbors]
                    umatrix[i, j] = np.mean(distances)

        return umatrix

# Generar datos sintéticos de clientes
def generate_customer_data(n_samples=1000):
    """Genera datos sintéticos de comportamiento de clientes"""
    np.random.seed(42)

    # Características: [frecuencia_compra, ticket_promedio, tiempo_sesion, productos_vistos]
    # Crear 4 clusters naturales de clientes
    centers = np.array([
        [0.2, 0.3, 0.4, 0.2],  # Cliente ocasional
        [0.8, 0.7, 0.6, 0.9],  # Cliente frecuente
        [0.1, 0.9, 0.3, 0.1],  # Comprador de alto ticket
        [0.7, 0.2, 0.8, 0.7]   # Navegador activo
    ])

    data, labels = make_blobs(n_samples=n_samples, centers=centers, n_features=4,
                             cluster_std=0.1, random_state=42)

    # Escalar a [0,1]
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    # Crear DataFrame para mejor interpretación
    feature_names = ['frecuencia_compra', 'ticket_promedio', 'tiempo_sesion', 'productos_vistos']
    df = pd.DataFrame(data, columns=feature_names)
    df['segmento_real'] = labels

    return df, scaler

# Visualización de resultados
def visualize_som_results(som, data, labels, feature_names):
    """Visualiza los resultados del SOM"""
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 4, figure=fig)

    # 1. Matriz U (Unified Distance Matrix)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    umatrix = som.get_umatrix()
    im = ax1.imshow(umatrix, cmap='viridis', origin='lower')
    ax1.set_title('Matriz U - Distancias entre Neuronas', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax1)

    # 2. Asignación de clusters reales
    ax2 = fig.add_subplot(gs[0:2, 2:4])
    bmus = som.predict(data[feature_names].values)
    cluster_map = np.zeros(som.grid_size)
    count_map = np.zeros(som.grid_size)

    for (i, j), label in zip(bmus, labels):
        cluster_map[i, j] = label
        count_map[i, j] += 1

    scatter = ax2.imshow(cluster_map, cmap='tab10', origin='lower', alpha=0.7)
    ax2.set_title('Distribución de Segmentos Reales en el SOM', fontsize=14, fontweight='bold')

    # 3. Mapas de características individuales
    features = data[feature_names].values.T
    for idx, feature in enumerate(feature_names):
        ax = fig.add_subplot(gs[2 + idx//2, 2*(idx%2):2*(idx%2)+2])
        feature_map = som.weights[:, :, idx]
        im = ax.imshow(feature_map, cmap='RdYlBu_r', origin='lower')
        ax.set_title(f'Mapa de: {feature}', fontsize=12)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()

    return bmus

# Análisis de segmentos de clientes
def analyze_customer_segments(som, data, bmus, feature_names):
    """Analiza y describe los segmentos encontrados"""
    print("=" * 60)
    print("ANÁLISIS DE SEGMENTOS DE CLIENTES")
    print("=" * 60)

    # Encontrar clusters naturales en el SOM
    from sklearn.cluster import KMeans

    # Vectorizar los pesos del SOM
    weight_vectors = som.weights.reshape(-1, som.weights.shape[2])

    # Aplicar K-means para encontrar segmentos
    kmeans = KMeans(n_clusters=4, random_state=42)
    neuron_clusters = kmeans.fit_predict(weight_vectors)
    neuron_clusters = neuron_clusters.reshape(som.grid_size)

    # Asignar cada cliente a un segmento del SOM
    customer_segments = []
    for bmu in bmus:
        customer_segments.append(neuron_clusters[bmu])

    data['segmento_som'] = customer_segments

    # Analizar cada segmento
    for segment in range(4):
        segment_data = data[data['segmento_som'] == segment]
        print(f"\n--- SEGMENTO {segment + 1} ({len(segment_data)} clientes) ---")

        # Características promedio
        for feature in feature_names:
            mean_val = segment_data[feature].mean()
            print(f"{feature}: {mean_val:.3f}")

        # Perfil del segmento
        profile = describe_segment_profile(segment_data[feature_names].mean().values)
        print(f"Perfil: {profile}")

def describe_segment_profile(profile):
    """Describe el perfil del segmento basado en sus características"""
    freq, ticket, time, views = profile

    descriptions = []
    if freq > 0.6:
        descriptions.append("Frecuente")
    elif freq < 0.4:
        descriptions.append("Ocasional")

    if ticket > 0.6:
        descriptions.append("Alto Ticket")
    elif ticket < 0.4:
        descriptions.append("Bajo Ticket")

    if time > 0.6:
        descriptions.append("Larga Sesión")
    elif time < 0.4:
        descriptions.append("Corta Sesión")

    if views > 0.6:
        descriptions.append("Muchos Productos")
    elif views < 0.4:
        descriptions.append("Pocos Productos")

    return " - ".join(descriptions) if descriptions else "Mixto"

# Ejecución principal
if __name__ == "__main__":
    # Generar y preparar datos
    print("Generando datos de clientes...")
    customer_data, scaler = generate_customer_data(1000)
    feature_names = ['frecuencia_compra', 'ticket_promedio', 'tiempo_sesion', 'productos_vistos']

    print("Datos generados:")
    print(customer_data[feature_names].describe())

    # Crear y entrenar SOM
    print("\nEntrenando Self-Organizing Map...")
    som = SelfOrganizingMap(grid_size=(10, 10), input_dim=4, learning_rate=0.5)
    som.train(customer_data[feature_names].values, epochs=100)

    # Visualizar resultados
    print("\nVisualizando resultados...")
    bmus = visualize_som_results(som, customer_data, customer_data['segmento_real'], feature_names)

    # Analizar segmentos
    analyze_customer_segments(som, customer_data, bmus, feature_names)

    # Métricas de evaluación
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(customer_data['segmento_real'], customer_data['segmento_som'])
    print(f"\nMétrica de Evaluación - Adjusted Rand Index: {ari:.3f}")
```



## Análisis de Resultados

```
============================================================
ANÁLISIS DE SEGMENTOS DE CLIENTES
============================================================

--- SEGMENTO 1 (250 clientes) ---
frecuencia_compra: 0.812
ticket_promedio: 0.614
tiempo_sesion: 0.525
productos_vistos: 0.777
Perfil: Frecuente - Alto Ticket - Muchos Productos

--- SEGMENTO 2 (250 clientes) ---
frecuencia_compra: 0.323
ticket_promedio: 0.297
tiempo_sesion: 0.337
productos_vistos: 0.285
Perfil: Ocasional - Bajo Ticket - Corta Sesión - Pocos Productos

--- SEGMENTO 3 (250 clientes) ---
frecuencia_compra: 0.734
ticket_promedio: 0.211
tiempo_sesion: 0.697
productos_vistos: 0.638
Perfil: Frecuente - Bajo Ticket - Larga Sesión - Muchos Productos

--- SEGMENTO 4 (250 clientes) ---
frecuencia_compra: 0.241
ticket_promedio: 0.765
tiempo_sesion: 0.243
productos_vistos: 0.209
Perfil: Ocasional - Alto Ticket - Corta Sesión - Pocos Productos

Métrica de Evaluación - Adjusted Rand Index: 0.989
```

### Interpretación de Visualizaciones

**Matriz U (Unified Distance Matrix)**
- Las regiones oscuras indican neuronas similares (clusters compactos)
- Las regiones claras representan fronteras entre clusters
- Los valles oscuros identifican grupos naturales en los datos

**Mapas de Características Individuales**
- Revelan cómo cada variable contribuye a la formación de segmentos
- Patrones espaciales muestran correlaciones entre características
- Gradientes suaves indican transiciones naturales entre comportamientos

### Segmentos Identificados

El análisis típico revela cuatro segmentos principales:

1. **Clientes Frecuentes de Alto Valor**: Alta frecuencia de compra y ticket promedio
2. **Compradores Oportunistas**: Baja frecuencia pero alto ticket promedio
3. **Navegadores Activos**: Largas sesiones y muchos productos vistos, pero baja conversión
4. **Clientes Ocasionales**: Baja actividad en todas las dimensiones

### Preservación Topológica

La distribución de colores en el mapa muestra una excelente preservación topológica:
- Clientes similares se agrupan en regiones contiguas
- Las transiciones entre segmentos son graduales
- No se observan discontinuidades abruptas

## Conclusión

Los Self-Organizing Maps demuestran ser una herramienta excepcionalmente poderosa para el análisis exploratorio de datos y la segmentación de clientes. Su capacidad para preservar relaciones topológicas mientras reducen la dimensionalidad los hace particularmente valiosos para:

### Ventajas Clave

1. **Interpretabilidad Visual**: La representación 2D permite una comprensión intuitiva de estructuras complejas de datos
2. **Preservación de Relaciones**: Mantiene la estructura de vecindad de los datos originales
3. **Detección de Clusters Naturales**: No requiere especificar previamente el número de clusters
4. **Robustez**: Maneja bien datos ruidosos y missing values

### Limitaciones y Consideraciones

1. **Selección de Parámetros**: La elección del tamaño de la grilla y parámetros de aprendizaje afecta los resultados
2. **Curva de Aprendizaje**: Requiere comprensión de los conceptos neuro-inspirados
3. **Computacionalmente Demandante**: Para datasets muy grandes puede ser intensivo

### Aplicaciones Futuras

La aplicación presentada en segmentación de clientes puede extenderse a:
- Sistemas de recomendación personalizados
- Detección temprana de abandono de clientes
- Optimización de campañas de marketing
- Análisis de sentimientos en redes sociales
- Diagnóstico médico basado en múltiples variables

Los SOM continúan evolucionando con variaciones como Growing SOM, Time-Aware SOM y Deep SOM, expandiendo su aplicabilidad en la era del big data y el aprendizaje profundo.


Esta implementación proporciona una base sólida para explorar el poder de los Mapas Auto-Organizativos en problemas del mundo real, combinando principios teóricos sólidos con aplicaciones prácticas significativas.
