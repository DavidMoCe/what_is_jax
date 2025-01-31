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
#### El código del ejemplo esta en el archivo [`ejemplo.py`](https://github.com/DavidMoCe/what_is_jax/blob/main/optimizaci%C3%B3n_simple_con_jax.py)
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
