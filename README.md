## 🌍 Chose Your Language / Elige tu idioma:
- [English](#english-gb)
- [Español](#español-es)

---

## English GB

# What is JAX?
**JAX** is a new Python library for machine learning developed by Google, designed for high-performance numerical computation, especially in machine learning and optimization. It is an enhanced alternative to NumPy, although its API for numerical functions is based on it.

The advantage of JAX is that it was conceived for both execution modes (Eager and Graph) from the beginning, avoiding the issues of its predecessors _PyTorch_ and _Tensorflow_. 

JAX is widely used in neural network models, such as in libraries built on top of it, like [**Flax**](https://flax.readthedocs.io/en/latest/) and [**Haiku**](https://dm-haiku.readthedocs.io/en/latest/). 

>[!NOTE]
>💡
>JAX might currently be the most advanced in terms of `Machine Learning` (ML) and promises to make machine learning programming more intuitive, structured, and clean. And most importantly, it can replace **Tensorflow** and **PyTorch** with significant advantages.


# Features
1. **Automatic Differentiation**: Efficiently computes derivatives of functions with `jax.grad()`, useful in optimization and deep learning. 🔧
2. **Just-In-Time (JIT) Compilation**: Uses `jax.jit()` to accelerate code by compiling it with **XLA** (Accelerated Linear Algebra). ⚡
3. **GPU/TPU Execution**:  Runs code on CPU, GPU, and TPU without modification. 💻🖥️
4. **Automatic Vectorization**: With `jax.vmap()`, applies operations to multiple data points in parallel. 🔄
5. **NumPy-like Operations**: Provides a very similar API to NumPy (`jax.numpy`), easing the transition. ➡️


# Comparison: JAX vs TensorFlow vs PyTorch

JAX, TensorFlow, and PyTorch are popular libraries for numerical computation and deep learning. This comparison highlights their key differences in performance, ease of use, and application.

## Comparison Table

| **Feature**       | **JAX** 🦎  | **TensorFlow** 🔵 | **PyTorch** 🔥 |
|-------------------------|------------|----------------|----------------|
| **Paradigm** | Functional, Stateless | Declarative (Graphs) | Imperative (Define-by-Run) |
| **Automatic Differentiation** | ✅ `jax.grad()` | ✅ `tf.GradientTape()` | ✅ `torch.autograd` |
| **JIT Compilation (Just-In-Time)** | ✅ `jax.jit()` (XLA) | ✅ `tf.function()` (XLA) | ❌ (Only in TorchScript) |
| **Runs on GPU/TPU** | ✅ Automatic | ✅ `tf.device()` | ✅ `cuda()` |
| **Automatic Vectorization** | ✅ `jax.vmap()` | ❌ Not native | ❌ Not native |
| **NumPy Usage** | ✅ `jax.numpy` | ❌ Not direct | ✅ Easy conversion |
| **Ecosystem & Pretrained Models** | ⛰️ Growing (Flax, Haiku) | 🚀 Extensive (TF Hub, Keras) | 🔥 Vast (Torch Hub) |
| **Learning Curve** | 🟠 Medium | 🔴 High | 🟢 Low |

# 🎯 Choosing the Best Option
- **JAX**: Ideal for **numerical optimizations, advanced differentiation, and GPU/TPU computation**. Used in research. 🌐
- **TensorFlow**: Best for **production deployment and scalable applications** en the cloud. ☁️
- **PyTorch**: Perfect for **apid prototyping, research, and ease of use**. ⚡


# 🌍Ecosystem
## 📌 Libraries Built on JAX

### **Deep Learning Frameworks**
- **[Flax](https://github.com/google/flax)** 🏗️ – Modular, similar to PyTorch Lightning.
- **[Haiku](https://github.com/deepmind/dm-haiku)** 🏔️ – From DeepMind, scope-based structure.
- **[Objax](https://github.com/google/objax)** 🎯 – Object-oriented approach for ML in production.
- **[Equinox](https://github.com/patrick-kidger/equinox)** 🌱 – Functional models, without scopes.

### **Optimization and Mathematics**
- **[Optax](https://github.com/deepmind/optax)** 🛠️ – Advanced optimization compatible with Flax and Haiku.
- **[Chex](https://github.com/deepmind/chex)** ✅ – Debugging and testing tools.

### **Reinforcement Learning and Graph Networks**
- **[RLax](https://github.com/deepmind/rlax)** 🏆 – RL algorithms compatible with JAX.
- **[Jraph](https://github.com/deepmind/jraph)** 🔗 – Graph neural networks (GNNs) in JAX.

### **Integrations and Additional Tools**
- **XLA** 🚀 – Just-In-Time compilation to accelerate execution on GPU/TPU.
- **`functools.partial()`** 🛠️ – Modularity and advanced functionality.

## 💡 Which to Choose?
- **For Deep Learning** → Flax or Haiku. 🧠
- **For Advanced Optimization** → Optax. 🔧
- **For Reinforcement Learning** → RLax. 🏅
- **For Graph Neural Networks** → Jraph. 🧬
- **For Debugging and Testing** → Chex. 🛠️


# Example
## Simple Optimization with JAX
#### The code for the example is in the file [`optimización_simple_con_jax.py`](https://github.com/DavidMoCe/what_is_jax/blob/main/optimizaci%C3%B3n_simple_con_jax.py)
Let's compute the minimum of a simple function, such as the **quadratic function** \( f(x) = x^2 + 3x + 2 \).

1. **Define the function**: We create the function whose derivative we want to compute. 📝
2. **Automatic Differentiation**: We use `jax.grad()` to compute the derivative of the function. 📐
3. **Optimization**: We use the derivative to perform a simple optimization (gradient descent). ⬇️

## Explanation:
- **Defining the function**:  
  \( f(x) = x^2 + 3x + 2 \) is a simple quadratic function.

- **Calculating the derivative**:  
  `jax.grad(func)` gives us the derivative of the function,  
  \( f'(x) = 2x + 3 \).

- **Optimization**:  
  We use **gradient descent**: we take the derivative value and use it to adjust the value of \( x \) in each iteration to minimize the function.

## Expected Output:

The output will show how the value of \( x \) changes with each iteration until it converges to the minimum of the function. The minimum of the function \( f(x) \) es \( x = -1.5 \).

# Example 2
## Wine Classification with JAX
#### The code for the example is in the file [`jax_calidad_del_vino.py`](https://github.com/DavidMoCe/what_is_jax/blob/main/jax_calidad_del_vino.py)
This project implements a neural network in JAX to classify wine quality using the `wine` dataset from `sklearn`.

## Main Steps:
1. **Data Loading and Preprocessing** 📊
   - The dataset is loaded using `sklearn.datasets.load_wine()`.
   - Data is normalized with `StandardScaler` to improve model performance.
   
2. **Parameter Initialization** 🎛️
   - We generate random weights and biases using `jax.random.normal`.
   - These are stored in a dictionary for use in the neural network.

3. **Model Definition** 🧠
   - A neural network with three layers is implemented:
     - Input layer: 16 neurons with ReLU activation.
     - Hidden layer: 8 neurons with ReLU activation.
     - Output layer: 3 neurons with Softmax activation.
   - `jax.numpy.dot` is used for matrix multiplications.

4. **Loss Function and Optimization** 🔧
   - `jax.nn.one_hot` is used to convert labels into vectors.
   - Cross-entropy is calculated to evaluate model error.
   - The `optax.adam` optimizer is used with a learning rate of `0.001`.

5. **Model Training** 🚀
   - Weights are updated using `grad(loss_fn)`.
   - The model is trained for 200 epochs.
   - Loss is displayed every 10 epochs.

6. **Model Evaluation** 📈
   - Accuracy is computed on the test set using `accuracy_score`.
   - A confusion matrix is visualized using `ConfusionMatrixDisplay`.

## Expected Output:
The trained model classifies wines with an accuracy above 90% on the test set.

# References
## Information on JAX
- [https://eiposgrados.com/blog-python/jax-machine-learning/](https://eiposgrados.com/blog-python/jax-machine-learning/)
- [https://es.eitca.org/inteligencia-artificial/eitc-ai-gcml-google-nube-aprendizaje-autom%C3%A1tico/plataforma-google-cloud-ai/introducci%C3%B3n-a-jax/revisi%C3%B3n-de-examen-introducci%C3%B3n-a-jax/%C2%BFCu%C3%A1les-son-las-caracter%C3%ADsticas-de-jax-que-permiten-el-m%C3%A1ximo-rendimiento-en-el-entorno-de-python%3F/](https://es.eitca.org/inteligencia-artificial/eitc-ai-gcml-google-nube-aprendizaje-autom%C3%A1tico/plataforma-google-cloud-ai/introducci%C3%B3n-a-jax/revisi%C3%B3n-de-examen-introducci%C3%B3n-a-jax/%C2%BFCu%C3%A1les-son-las-caracter%C3%ADsticas-de-jax-que-permiten-el-m%C3%A1ximo-rendimiento-en-el-entorno-de-python%3F/)

## TensorFlow and PyTorch
- [https://www.tensorflow.org/?hl=es-419](https://www.tensorflow.org/?hl=es-419)
- [https://pytorch.org/](https://pytorch.org/)

## Flax and Haiku
- [https://flax.readthedocs.io/en/latest/](https://flax.readthedocs.io/en/latest/)
- [https://dm-haiku.readthedocs.io/en/latest/](https://dm-haiku.readthedocs.io/en/latest/)

## Comparison between JAX, TensorFlow, and PyTorch
- [https://www.computerworld.es/article/2115282/tensorflow-pytorch-y-jax-los-principales-marcos-de-deep-learning.html](https://www.computerworld.es/article/2115282/tensorflow-pytorch-y-jax-los-principales-marcos-de-deep-learning.html)

## Collecting and Structuring Information
- [https://chatgpt.com/](https://chatgpt.com/)

# 📃 License
This project is under the MIT License. See the [`LICENSE`](https://github.com/DavidMoCe/what_is_jax/blob/main/LICENSE.TXT) file for more details.

---

## Español ES

# ¿Qué es JAX?
**JAX** es una nueva biblioteca de Python de aprendizaje automático de Google, diseñada para la computación numérica de alto rendimiento, especialmente en aprendizaje automático y optimización. Es una alternativa mejorada a NumPy aunque su API para funciones numéricas se basa en esta.

La ventaja de JAX es que fue concebido para ambos modos de ejecución (Eager y Graph) desde el principio y adolece de los problemas de sus predecesores _PyTorch_ y _Tensorflow_. 

JAX es ampliamente utilizado en modelos de redes neuronales, como en bibliotecas construidas sobre él, como [**Flax**](https://flax.readthedocs.io/en/latest/) y [**Haiku**](https://dm-haiku.readthedocs.io/en/latest/). 

>[!NOTE]
>💡
>JAX quizás sea actualmente lo más avanzado en términos `Machine Learning` (ML) y promete hacer que la programación de aprendizaje automático sea más intuitiva, estructurada y limpia. Y, sobre todo, puede reemplazar con importantes ventajas a **Tensorflow** y **PyTorch**.


# Características
1. **Diferenciación automática**: Permite calcular derivadas de funciones de manera eficiente con `jax.grad()`, útil en optimización y aprendizaje profundo. 🔧
2. **Compilación Just-In-Time (JIT)**: Usa `jax.jit()` para acelerar el código al compilarlo con **XLA** (Accelerated Linear Algebra). ⚡
3. **Ejecución en GPU/TPU**: Puede ejecutar código en CPU, GPU y TPU sin necesidad de cambios. 💻🖥️
4. **Vectorización automática**: Con `jax.vmap()`, permite aplicar operaciones a múltiples datos en paralelo. 🔄
5. **Operaciones similares a NumPy**: Proporciona una API muy parecida a NumPy (`jax.numpy`), facilitando la transición. ➡️


# Comparación: JAX vs TensorFlow vs PyTorch

JAX, TensorFlow y PyTorch son bibliotecas populares para cómputo numérico y aprendizaje profundo. Esta comparación destaca sus diferencias clave en rendimiento, facilidad de uso y aplicación.

## Tabla Comparativa

| **Característica**       | **JAX** 🦎  | **TensorFlow** 🔵 | **PyTorch** 🔥 |
|-------------------------|------------|----------------|----------------|
| **Paradigma** | Funcional, sin estado | Declarativo (Graphs) | Imperativo (Define-by-Run) |
| **Diferenciación Automática** | ✅ `jax.grad()` | ✅ `tf.GradientTape()` | ✅ `torch.autograd` |
| **Compilación JIT (Just-In-Time)** | ✅ `jax.jit()` (XLA) | ✅ `tf.function()` (XLA) | ❌ (Solo en TorchScript) |
| **Ejecuta en GPU/TPU** | ✅ Automático | ✅ `tf.device()` | ✅ `cuda()` |
| **Vectorización Automática** | ✅ `jax.vmap()` | ❌ No nativo | ❌ No nativo |
| **Uso de NumPy** | ✅ `jax.numpy` | ❌ No directo | ✅ Conversión fácil |
| **Ecosistema y Modelos Preentrenados** | ⛰️ Creciendo (Flax, Haiku) | 🚀 Amplio (TF Hub, Keras) | 🔥 Extenso (Torch Hub) |
| **Curva de Aprendizaje** | 🟠 Media | 🔴 Alta | 🟢 Baja |

# 🎯 Elección de la Mejor Opción
- **JAX**: Ideal para **optimizaciones numéricas, diferenciación avanzada y computación en GPU/TPU**. Usado en investigación. 🌐
- **TensorFlow**: Mejor para **despliegue en producción y aplicaciones escalables** en la nube. ☁️
- **PyTorch**: Perfecto para **prototipado rápido, investigación y facilidad de uso**. ⚡


# 🌍Ecosistema
## 📌 Librerías Implementadas sobre JAX

### **Frameworks de Deep Learning**
- **[Flax](https://github.com/google/flax)** 🏗️ – Modular, similar a PyTorch Lightning.
- **[Haiku](https://github.com/deepmind/dm-haiku)** 🏔️ – De DeepMind, estructura basada en scopes.
- **[Objax](https://github.com/google/objax)** 🎯 – Enfoque orientado a objetos para ML en producción.
- **[Equinox](https://github.com/patrick-kidger/equinox)** 🌱 – Modelos funcionales, sin scopes.

### **Optimización y Matemáticas**
- **[Optax](https://github.com/deepmind/optax)** 🛠️ – Optimización avanzada compatible con Flax y Haiku.
- **[Chex](https://github.com/deepmind/chex)** ✅ – Herramientas de depuración y testing.

### **Aprendizaje por Refuerzo y Redes Gráficas**
- **[RLax](https://github.com/deepmind/rlax)** 🏆 – Algoritmos de RL compatibles con JAX.
- **[Jraph](https://github.com/deepmind/jraph)** 🔗 – Redes neuronales gráficas (GNNs) en JAX.

### **Integraciones y Herramientas Adicionales**
- **XLA** 🚀 – Compilación Just-In-Time para acelerar ejecución en GPU/TPU.
- **`functools.partial()`** 🛠️ – Modularidad y funcionalidad avanzada.

## 💡 ¿Cuál Elegir?
- **Para Deep Learning** → Flax o Haiku. 🧠
- **Para Optimización Avanzada** → Optax. 🔧
- **Para Aprendizaje por Refuerzo** → RLax. 🏅
- **Para Redes Neuronales Gráficas** → Jraph. 🧬
- **Para Depuración y Pruebas** → Chex. 🛠️


# Ejemplo
## Optimización simple con JAX
#### El código del ejemplo esta en el archivo [`optimización_simple_con_jax.py`](https://github.com/DavidMoCe/what_is_jax/blob/main/optimizaci%C3%B3n_simple_con_jax.py)
Vamos a calcular el mínimo de una función simple, como la **función cuadrática** \( f(x) = x^2 + 3x + 2 \).

1. **Definir la función**: Creamos la función cuya derivada queremos calcular. 📝
2. **Diferenciación Automática**: Usamos `jax.grad()` para calcular la derivada de la función. 📐
3. **Optimización**: Usamos la derivada para hacer una optimización simple (descenso por gradiente). ⬇️

## Explicación:
- **Definición de la función**:  
  \( f(x) = x^2 + 3x + 2 \) es una función cuadrática simple.

- **Cálculo de la derivada**:  
  `jax.grad(func)` nos da la derivada de la función, es decir,  
  \( f'(x) = 2x + 3 \).

- **Optimización**:  
  Usamos **descenso por gradiente**: tomamos el valor de la derivada y lo usamos para ajustar el valor de \( x \) en cada iteración, con el fin de minimizar la función.

## Resultado esperado:

La salida te mostrará cómo el valor de \( x \) cambia en cada iteración hasta converger al mínimo de la función. El mínimo de la función \( f(x) \) es \( x = -1.5 \).

# Ejemplo 2
## Clasificación de Vinos con JAX
#### El código del ejemplo esta en el archivo [`jax_calidad_del_vino.py`](https://github.com/DavidMoCe/what_is_jax/blob/main/jax_calidad_del_vino.py)
Este proyecto implementa una red neuronal en JAX para clasificar la calidad del vino utilizando el conjunto de datos `wine` de `sklearn`.

## Pasos que realiza el proyecto:
1. Carga y preprocesamiento de datos 📊
    - Se carga el dataset de `sklearn.datasets.load_wine()`.
    - Se normalizan los datos con `StandardScaler` para mejorar el rendimiento del modelo.

2. Inicialización de parámetros 🎛️
    - Se generan pesos y sesgos aleatorios usando `jax.random.normal`.
    - Se almacenan en un diccionario para su uso en la red neuronal.

3. Definición del modelo 🧠
    - Se implementa una red neuronal con tres capas:
       - Capa de entrada: 16 neuronas y activación ReLU.
       - Capa oculta: 8 neuronas y activación ReLU.
       - Capa de salida: 3 neuronas con activación Softmax.
    - Se usa `jax.numpy.dot` para las multiplicaciones de matrices.

4. Función de pérdida y optimización 🔧
    - Se usa `jax.nn.one_hot` para convertir etiquetas en vectores.
    - Se calcula la entropía cruzada para evaluar el error del modelo.
    - Se usa el optimizador `optax.adam` con una tasa de aprendizaje de `0.001`.

5. Entrenamiento del modelo 🚀
    - Se actualizan los pesos usando `grad(loss_fn)`.
    - Se entrena durante 200 épocas.
    - Se muestra la pérdida cada 10 épocas.

6. Evaluación del modelo 📈
    - Se calcula la precisión en el conjunto de prueba con `accuracy_score`.
    - Se visualiza la matriz de confusión con `ConfusionMatrixDisplay`.

## Resultado esperado:
El modelo entrenado clasifica los vinos con una precisión superior al 90% en el conjunto de prueba.

# Bibliografía
## Información sobre JAX
- [https://eiposgrados.com/blog-python/jax-machine-learning/](https://eiposgrados.com/blog-python/jax-machine-learning/)
- [https://es.eitca.org/inteligencia-artificial/eitc-ai-gcml-google-nube-aprendizaje-autom%C3%A1tico/plataforma-google-cloud-ai/introducci%C3%B3n-a-jax/revisi%C3%B3n-de-examen-introducci%C3%B3n-a-jax/%C2%BFCu%C3%A1les-son-las-caracter%C3%ADsticas-de-jax-que-permiten-el-m%C3%A1ximo-rendimiento-en-el-entorno-de-python%3F/](https://es.eitca.org/inteligencia-artificial/eitc-ai-gcml-google-nube-aprendizaje-autom%C3%A1tico/plataforma-google-cloud-ai/introducci%C3%B3n-a-jax/revisi%C3%B3n-de-examen-introducci%C3%B3n-a-jax/%C2%BFCu%C3%A1les-son-las-caracter%C3%ADsticas-de-jax-que-permiten-el-m%C3%A1ximo-rendimiento-en-el-entorno-de-python%3F/)

## TensorFlow y PyTorch
- [https://www.tensorflow.org/?hl=es-419](https://www.tensorflow.org/?hl=es-419)
- [https://pytorch.org/](https://pytorch.org/)

## Flax y Haiku
- [https://flax.readthedocs.io/en/latest/](https://flax.readthedocs.io/en/latest/)
- [https://dm-haiku.readthedocs.io/en/latest/](https://dm-haiku.readthedocs.io/en/latest/)

## Comparativa entre JAX, TensorFlow y PyTorch
- [https://www.computerworld.es/article/2115282/tensorflow-pytorch-y-jax-los-principales-marcos-de-deep-learning.html](https://www.computerworld.es/article/2115282/tensorflow-pytorch-y-jax-los-principales-marcos-de-deep-learning.html)

## Recopilar y estructurar la información
- [https://chatgpt.com/](https://chatgpt.com/)

# 📃 Licencia
Este proyecto está bajo la licencia MIT. Consulta el archivo [`LICENSE`](https://github.com/DavidMoCe/what_is_jax/blob/main/LICENSE.TXT) para más detalles.

