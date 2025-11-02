# Algoritmo de Eigenfaces para Reconocimiento Facial

## Introducción Teórica

El método de Eigenfaces, desarrollado por Turk y Pentland en 1991, es una técnica fundamental en el campo de reconocimiento facial que utiliza el Análisis de Componentes Principales (PCA). Este enfoque transforma el problema de reconocimiento facial en un problema de reducción de dimensionalidad en el espacio de características.

**Conceptos clave:**
- Las imágenes de rostros se representan como vectores en un espacio de alta dimensionalidad
- PCA identifica las direcciones de máxima varianza (eigenfaces) en este espacio
- Cada rostro puede aproximarse como una combinación lineal de estas eigenfaces
- El reconocimiento se realiza comparando los pesos (coeficientes) en el espacio reducido

## Fórmulas Principales

### 1. Preprocesamiento y Cálculo de la Media

Sea \( \Gamma_1, \Gamma_2, ..., \Gamma_M \) un conjunto de M imágenes de entrenamiento, cada una representada como un vector de dimensión \( N^2 \):

**Cara media:**
\[
\Psi = \frac{1}{M} \sum_{i=1}^{M} \Gamma_i
\]

**Vectores de diferencia (rostros centrados):**
\[
\Phi_i = \Gamma_i - \Psi \quad \text{para } i = 1, 2, ..., M
\]

### 2. Matriz de Covarianza y Diagonalización

**Matriz de covarianza:**
\[
C = \frac{1}{M} \sum_{i=1}^{M} \Phi_i \Phi_i^T = AA^T
\]
donde \( A = [\Phi_1, \Phi_2, ..., \Phi_M] \) es una matriz \( N^2 \times M \)

**Problema de eigenvalores:**
\[
C u_k = \lambda_k u_k
\]
donde \( u_k \) son los eigenvectores (eigenfaces) y \( \lambda_k \) los eigenvalores correspondientes

### 3. Solución Computacionalmente Eficiente

Dado que \( C \) es muy grande (\( N^2 \times N^2 \)), se utiliza el truco de Turk y Pentland:

Sea \( L = A^T A \) (matriz \( M \times M \)), resolvemos:
\[
L v_i = \mu_i v_i
\]

Los eigenvectores de \( C \) se obtienen como:
\[
u_i = A v_i
\]

### 4. Proyección y Reconstrucción

**Proyección en el espacio de características:**
Para una nueva imagen \( \Gamma \):
\[
\omega_k = u_k^T (\Gamma - \Psi)
\]
Vector de características:
\[
\Omega = [\omega_1, \omega_2, ..., \omega_K]^T
\]

**Reconstrucción de la imagen:**
\[
\Gamma_{reconstruida} = \Psi + \sum_{k=1}^{K} \omega_k u_k
\]

### 5. Medidas de Distancia para Reconocimiento

**Distancia euclidiana en el espacio de características:**
\[
\epsilon_k = \|\Omega - \Omega_k\|^2
\]

**Umbral de reconocimiento:**
\[
\theta = \max \|\Omega - \Omega_k\| \quad \text{para todas las imágenes de entrenamiento}
\]

## Implementación en Python

```python
# =============================================================================
# BLOQUE 1: IMPORTACIONES Y CONFIGURACIÓN
# =============================================================================

import numpy as np
import cv2
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter

# =============================================================================
# BLOQUE 2: CLASE PRINCIPAL EIGENFACES
# =============================================================================

class EigenFaces:
    def __init__(self, n_components=15, img_size=(100, 100)):
        """
        Inicializa el modelo Eigenfaces
        
        Args:
            n_components: Número de componentes principales (se ajusta automáticamente)
            img_size: Tamaño al que se redimensionarán las imágenes
        """
        self.n_components = n_components
        self.img_size = img_size
        self.pca = None
        self.mean_face = None
        self.training_labels = []
        self.training_features = None
        self.explained_variance = None
        
    # =========================================================================
    # BLOQUE 3: CARGA Y PREPROCESAMIENTO DE IMÁGENES
    # =========================================================================
    
    def cargar_dataset(self, carpeta_dataset):
        """
        Carga todas las imágenes desde la estructura de carpetas
        
        Estructura esperada:
        dataset/
        ├── persona1/
        │   ├── imagen1.jpg
        │   ├── imagen2.jpg
        │   └── ...
        ├── persona2/
        │   ├── imagen1.jpg
        │   └── ...
        └── ...
        
        Returns:
            X: Array de imágenes aplanadas
            labels: Lista de etiquetas
        """
        print("=" * 50)
        print("CARGANDO DATASET")
        print("=" * 50)
        
        images = []
        labels = []
        conteo_personas = {}
        
        # Verificar que la carpeta existe
        if not os.path.exists(carpeta_dataset):
            raise ValueError(f"La carpeta '{carpeta_dataset}' no existe")
        
        # Recorrer todas las subcarpetas (cada persona)
        for persona in sorted(os.listdir(carpeta_dataset)):
            persona_path = os.path.join(carpeta_dataset, persona)
            
            if os.path.isdir(persona_path):
                print(f"Cargando imágenes de: {persona}")
                imagenes_persona = 0
                
                # Cargar todas las imágenes de la persona
                for archivo in sorted(os.listdir(persona_path)):
                    if archivo.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = os.path.join(persona_path, archivo)
                        
                        # Cargar y preprocesar imagen
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            # Redimensionar y normalizar
                            img_resized = cv2.resize(img, self.img_size)
                            img_normalized = img_resized.astype(np.float32) / 255.0
                            
                            images.append(img_normalized.flatten())
                            labels.append(persona)
                            imagenes_persona += 1
                
                conteo_personas[persona] = imagenes_persona
                print(f"--- {imagenes_persona} imágenes cargadas")
        
        # Resumen del dataset
        total_imagenes = len(images)
        total_personas = len(conteo_personas)
        
        print("\nRESUMEN DEL DATASET:")
        print(f"   Total de personas: {total_personas}")
        print(f"   Total de imágenes: {total_imagenes}")
        for persona, count in conteo_personas.items():
            print(f"------ {persona} : {count} imágenes")
        
        if total_imagenes == 0:
            raise ValueError("No se encontraron imágenes válidas en el dataset")
        
        return np.array(images), labels
    
    # =========================================================================
    # BLOQUE 4: ENTRENAMIENTO DEL MODELO
    # =========================================================================
    
    def entrenar(self, carpeta_dataset):
        """
        Entrena el modelo Eigenfaces con el dataset
        
        Args:
            carpeta_dataset: Ruta a la carpeta con las imágenes de entrenamiento
        """
        print("\n" + "=" * 50)
        print("ENTRENANDO MODELO")
        print("=" * 50)
        
        # Cargar dataset
        X, self.training_labels = self.cargar_dataset(carpeta_dataset)
        n_muestras = X.shape[0]
        
        # Ajustar automáticamente el número de componentes
        self.n_components = min(self.n_components, n_muestras - 1)
        print(f"Configuración: {self.n_components} componentes para {n_muestras} imágenes")
        
        # Calcular cara media
        print("Calculando cara media...")
        self.mean_face = np.mean(X, axis=0)
        
        # Centrar los datos
        X_centered = X - self.mean_face
        
        # Aplicar PCA
        print("Aplicando Análisis de Componentes Principales...")
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_centered)
        
        # Transformar datos de entrenamiento
        self.training_features = self.pca.transform(X_centered)
        self.explained_variance = self.pca.explained_variance_ratio_
        
        # Mostrar información del entrenamiento
        varianza_total = np.sum(self.explained_variance)
        print(f"Entrenamiento completado")
        print(f"Varianza explicada total: {varianza_total:.2%}")
        print(f"Componentes utilizadas: {self.n_components}")
        
        return self
    
    # =========================================================================
    # BLOQUE 5: RECONOCIMIENTO DE IMÁGENES
    # =========================================================================
    
    def preprocesar_imagen(self, imagen_path):
        """
        Preprocesa una imagen individual para reconocimiento
        
        Args:
            imagen_path: Ruta a la imagen a preprocesar
            
        Returns:
            Imagen aplanada y preprocesada
        """
        img = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {imagen_path}")
        
        img_resized = cv2.resize(img, self.img_size)
        img_normalized = img_resized.astype(np.float32) / 255.0
        return img_normalized.flatten()
    
    def reconocer(self, imagen_path, umbral=0.8):
        """
        Reconoce una imagen y devuelve la persona identificada
        
        Args:
            imagen_path: Ruta a la imagen a reconocer
            umbral: Umbral de confianza (0-1)
            
        Returns:
            dict: Resultados del reconocimiento
        """
        if self.pca is None:
            raise ValueError("El modelo debe ser entrenado primero")
        
        try:
            # Preprocesar imagen
            img_flat = self.preprocesar_imagen(imagen_path)
            
            # Proyectar en el espacio de características
            img_centered = img_flat - self.mean_face
            features_prueba = self.pca.transform([img_centered])[0]
            
            # Calcular similitudes (usando distancia coseno)
            similitudes = []
            for features in self.training_features:
                # Distancia coseno (convertida a similitud)
                sim = np.dot(features_prueba, features) / (
                    np.linalg.norm(features_prueba) * np.linalg.norm(features)
                )
                similitudes.append(sim)
            
            # Encontrar la mejor coincidencia
            max_sim = np.max(similitudes)
            best_idx = np.argmax(similitudes)
            persona = self.training_labels[best_idx]
            
            # Calcular confianza normalizada
            confianza = float(max_sim)
            
            # Determinar resultado basado en el umbral
            if confianza >= umbral:
                resultado = "RECONOCIDO"
            else:
                resultado = "DESCONOCIDO"
                persona = "Desconocido"
            
            return {
                'archivo': os.path.basename(imagen_path),
                'persona': persona,
                'confianza': confianza,
                'resultado': resultado,
                'umbral_utilizado': umbral
            }
            
        except Exception as e:
            return {
                'archivo': os.path.basename(imagen_path),
                'persona': 'ERROR',
                'confianza': 0.0,
                'resultado': f'Error: {str(e)}',
                'umbral_utilizado': umbral
            }
    
    def probar_carpeta(self, carpeta_pruebas, umbral=0.8):
        """
        Prueba todas las imágenes en una carpeta de pruebas
        
        Args:
            carpeta_pruebas: Ruta a la carpeta con imágenes de prueba
            umbral: Umbral de confianza para reconocimiento
        """
        print("\n" + "=" * 50)
        print("PROBANDO IMÁGENES")
        print("=" * 50)
        
        if not os.path.exists(carpeta_pruebas):
            print(f"La carpeta '{carpeta_pruebas}' no existe")
            return []
        
        resultados = []
        archivos = [f for f in os.listdir(carpeta_pruebas) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not archivos:
            print("No se encontraron imágenes en la carpeta de pruebas")
            return []
        
        print(f"Encontradas {len(archivos)} imágenes para probar")
        
        for archivo in sorted(archivos):
            imagen_path = os.path.join(carpeta_pruebas, archivo)
            resultado = self.reconocer(imagen_path, umbral)
            resultados.append(resultado)
            
            # Mostrar resultado
            emoji = "✓" if resultado['resultado'] == 'RECONOCIDO' else "✕"
            print(f"{emoji} {archivo} -> {resultado['persona']} (conf: {resultado['confianza']:.3f})")
        
        return resultados
    
    # =========================================================================
    # BLOQUE 6: VISUALIZACIÓN Y ANÁLISIS
    # =========================================================================
    
    def visualizar_modelo(self):
        """Visualiza la cara media y las eigenfaces principales"""
        if self.pca is None:
            print("El modelo debe ser entrenado primero")
            return
        
        print("\n" + "=" * 50)
        print("VISUALIZANDO MODELO")
        print("=" * 50)
        
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.flatten()
        
        # Cara media
        axes[0].imshow(self.mean_face.reshape(self.img_size), cmap='gray')
        axes[0].set_title('Cara Media', fontweight='bold')
        axes[0].axis('off')
        
        # Primeras 3 eigenfaces
        for i in range(3):
            if i < len(self.pca.components_):
                eigenface = self.pca.components_[i].reshape(self.img_size)
                axes[i+1].imshow(eigenface, cmap='jet')
                axes[i+1].set_title(f'Eigenface {i+1}\n(Var: {self.explained_variance[i]:.4f})')
                axes[i+1].axis('off')
        
        # Varianza explicada acumulada
        varianza_acumulada = np.cumsum(self.explained_variance)
        axes[4].plot(range(1, len(varianza_acumulada) + 1), varianza_acumulada, 'b-o', linewidth=2)
        axes[4].axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% varianza')
        axes[4].axhline(y=0.90, color='g', linestyle='--', alpha=0.7, label='90% varianza')
        axes[4].set_xlabel('Número de Componentes')
        axes[4].set_ylabel('Varianza Explicada Acumulada')
        axes[4].set_title('Varianza Explicada', fontweight='bold')
        axes[4].grid(True, alpha=0.3)
        axes[4].legend()
        axes[4].set_ylim(0, 1.1)
        
        # Distribución de personas en el dataset
        conteo = Counter(self.training_labels)
        personas = list(conteo.keys())
        counts = list(conteo.values())
        
        axes[5].barh(personas, counts, color='skyblue')
        axes[5].set_xlabel('Número de Imágenes')
        axes[5].set_title('Distribución del Dataset', fontweight='bold')
        axes[5].grid(True, alpha=0.3, axis='x')
        
        # Espacio en blanco para los últimos 2 subplots
        for i in range(6, 8):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def mostrar_estadisticas(self):
        """Muestra estadísticas del modelo entrenado"""
        if self.pca is None:
            print("El modelo debe ser entrenado primero")
            return
        
        print("\n" + "=" * 50)
        print("ESTADÍSTICAS DEL MODELO")
        print("=" * 50)
        
        n_personas = len(set(self.training_labels))
        n_imagenes = len(self.training_labels)
        
        print(f"Personas en el modelo: {n_personas}")
        print(f"Imágenes de entrenamiento: {n_imagenes}")
        print(f"Componentes principales: {self.n_components}")
        print(f"Varianza explicada: {np.sum(self.explained_variance):.2%}")
        print(f" Tamaño de imagen: {self.img_size}")
        
        # Mostrar varianza por componentes
        print(f"\nVarianza por componentes principales:")
        for i, var in enumerate(self.explained_variance[:5]):  # Primeras 5
            print(f"--- Componente {i+1}: {var:.3%}")
        if len(self.explained_variance) > 5:
            print(f"   ... y {len(self.explained_variance) - 5} componentes más")

# =============================================================================
# BLOQUE 7: FUNCIÓN PRINCIPAL Y USO
# =============================================================================

def main():
    """
    Función principal de demostración
    """
    print("ALGORITMO EIGENFACES - RECONOCIMIENTO FACIAL")
    print("=" * 60)
    
    try:
        # 1. INICIALIZAR MODELO
        modelo = EigenFaces(
            n_components=20,      # Máximo de componentes (se ajustará automáticamente)
            img_size=(100, 100)   # Tamaño de las imágenes
        )
        
        # 2. ENTRENAR CON DATASET
        modelo.entrenar('dataset')
        
        # 3. MOSTRAR ESTADÍSTICAS
        modelo.mostrar_estadisticas()
        
        # 4. VISUALIZAR COMPONENTES
        modelo.visualizar_modelo()
        
        # 5. PROBAR IMÁGENES (si existe la carpeta de pruebas)
        if os.path.exists('test_images'):
            print("\n" + "=" * 60)
            print("INICIANDO PRUEBAS DE RECONOCIMIENTO")
            print("=" * 60)
            
            resultados = modelo.probar_carpeta(
                carpeta_pruebas='test_images',
                umbral=0.7  # Ajustar este valor (0-1): más alto = más estricto
            )
            
            # Resumen de pruebas
            if resultados:
                reconocidos = sum(1 for r in resultados if r['resultado'] == 'RECONOCIDO')
                total = len(resultados)
                print(f"\nRESUMEN DE PRUEBAS: {reconocidos}/{total} reconocidos")
        
        else:
            print("\nTip: Crea una carpeta 'test_images' para probar el reconocimiento")
        
        print("\nProceso completado exitosamente")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nSolución: Asegúrate de que:")
        print("   - La carpeta 'dataset' existe y tiene subcarpetas con imágenes")
        print("   - Las imágenes son válidas (JPG, PNG, etc.)")
        print("   - Cada persona tiene su propia subcarpeta dentro de 'dataset'")

if __name__ == "__main__":
    main()
```
## Resultado de ejecución

```bash
ALGORITMO EIGENFACES - RECONOCIMIENTO FACIAL
============================================================

==================================================
ENTRENANDO MODELO
==================================================
==================================================
CARGANDO DATASET
==================================================
Cargando imágenes de: .ipynb_checkpoints
--- 0 imágenes cargadas
Cargando imágenes de: persona1
--- 4 imágenes cargadas
Cargando imágenes de: persona2
--- 4 imágenes cargadas
Cargando imágenes de: persona3
--- 4 imágenes cargadas

RESUMEN DEL DATASET:
   Total de personas: 4
   Total de imágenes: 12
------ .ipynb_checkpoints : 0 imágenes
------ persona1 : 4 imágenes
------ persona2 : 4 imágenes
------ persona3 : 4 imágenes
Configuración: 11 componentes para 12 imágenes
Calculando cara media...
Aplicando Análisis de Componentes Principales...
Entrenamiento completado
Varianza explicada total: 100.00%
Componentes utilizadas: 11

==================================================
ESTADÍSTICAS DEL MODELO
==================================================
Personas en el modelo: 3
Imágenes de entrenamiento: 12
Componentes principales: 11
Varianza explicada: 100.00%
 Tamaño de imagen: (100, 100)

Varianza por componentes principales:
--- Componente 1: 51.988%
--- Componente 2: 31.396%
--- Componente 3: 4.040%
--- Componente 4: 2.815%
--- Componente 5: 2.294%
   ... y 6 componentes más

==================================================
VISUALIZANDO MODELO
==================================================


============================================================
INICIANDO PRUEBAS DE RECONOCIMIENTO
============================================================

==================================================
PROBANDO IMÁGENES
==================================================
Encontradas 4 imágenes para probar
✓ prueba1.jpeg -> persona1 (conf: 0.953)
✓ prueba2.jpeg -> persona2 (conf: 0.970)
✓ prueba3.jpeg -> persona3 (conf: 0.986)
✕ prueba4.jpeg -> Desconocido (conf: 0.606)

RESUMEN DE PRUEBAS: 3/4 reconocidos

Proceso completado exitosamente
```



# Análisis de Resultados del Sistema de Eigenfaces

## Resumen del Dataset y Configuración del Modelo

El experimento se realizó con un dataset compuesto por **4 personas diferentes**, conteniendo un total de **12 imágenes** distribuidas de manera equilibrada (4 imágenes por cada una de las 3 personas utilizadas en el entrenamiento). El modelo fue configurado para utilizar **11 componentes principales**, que es el número máximo posible dado que se tenían 12 imágenes de entrenamiento (generalmente se usa n-1 componentes para n imágenes). La varianza explicada alcanzó el **100%**, lo que indica que el modelo capturó toda la variabilidad presente en el conjunto de datos de entrenamiento. El tamaño de imagen procesado fue de 100x100 píxeles, lo que significa que cada imagen original fue representada en un espacio de 10,000 dimensiones antes de la reducción mediante PCA.

## Distribución de Varianza y Componentes Principales

El análisis de varianza reveló una distribución muy concentrada en las primeras componentes. La **primera componente principal** capturó el **51.988%** de la varianza total, mientras que la **segunda componente** explicó el **31.396%**. En conjunto, estas dos componentes representan más del **83% de la variabilidad total** del dataset. Esta concentración sugiere que las características faciales más distintivas en este dataset particular pueden ser capturadas eficientemente con muy pocas dimensiones. Las componentes restantes (3 a 11) explican proporciones decrecientes de varianza, desde 4.040% hasta valores menores, indicando que capturan características faciales más sutiles o específicas.

## Resultados de las Pruebas de Reconocimiento

### Pruebas con Personas Conocidas (1-3)
Las tres primeras pruebas demostraron un **reconocimiento excelente** con niveles de confianza muy altos:
- **Prueba 1**: Reconocida como "persona1" con **95.3%** de confianza
- **Prueba 2**: Reconocida como "persona2" con **97.0%** de confianza  
- **Prueba 3**: Reconocida como "persona3" con **98.6%** de confianza

Estos resultados indican que el modelo generalizó bien a nuevas imágenes de las personas conocidas, a pesar de que estas imágenes no formaban parte del conjunto de entrenamiento original. Los altos niveles de confianza (todos superiores al 95%) sugieren que las representaciones en el espacio de eigenfaces fueron robustas y distintivas para cada individuo.

### Prueba 4: Caso de Persona Desconocida
La **cuarta prueba** involucró una imagen de una **persona que no estaba incluida en el dataset de entrenamiento**. El sistema correctamente la clasificó como **"Desconocido"** con un nivel de confianza del **60.6%**. Este resultado es particularmente significativo porque:

- **Umbral de decisión efectivo**: La confianza del 60.6% está muy por debajo de los niveles de las personas conocidas (95.3%-98.6%), lo que sugiere que el modelo estableció un umbral de reconocimiento apropiado
- **Detección de outliers**: El sistema fue capaz de identificar correctamente que la cara no pertenecía a ninguna de las personas registradas
- **Margen claro**: La diferencia de aproximadamente 35 puntos porcentuales entre la confianza más baja de las personas conocidas y la confianza de la persona desconocida indica una buena separación entre clases

## Interpretación de las Métricas Clave

### Varianza Explicada (100%)
La varianza explicada del 100% indica que el modelo utiliza suficientes componentes para reconstruir perfectamente cualquier imagen del conjunto de entrenamiento. Sin embargo, esto podría llevar a sobreajuste si el dataset es pequeño.

### Niveles de Confianza
Los niveles de confianza representan qué tan "cerca" está la imagen proyectada en el espacio facial de la representación más similar en la base de datos. Un valor alto (>90%) indica alta similitud, mientras que valores medios (60%) sugieren que la cara comparte algunas características pero no coincide con ninguna identidad conocida.

### Componentes vs. Imágenes
El uso de 11 componentes para 12 imágenes sigue la práctica estándar en PCA facial, donde típicamente se usan n-1 componentes para n imágenes, evitando problemas de singularidad en la matriz de covarianza.

## Efectividad Global del Sistema

El sistema alcanzó una **precisión del 75%** en las pruebas (3 de 4 correctas), pero más importante aún, demostró **100% de efectividad en la tarea de verificación**: todas las personas conocidas fueron correctamente identificadas y la persona desconocida fue correctamente rechazada. Esto es crucial para aplicaciones de seguridad donde el rechazo correcto de intrusos es tan importante como el reconocimiento de usuarios autorizados.


