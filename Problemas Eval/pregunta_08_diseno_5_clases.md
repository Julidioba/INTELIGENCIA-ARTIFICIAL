# Pregunta 8 — Diseño: 5 clases de animales

## Enunciado
Clasificación de: *ranas, arañas, monos, ballenas, pájaros*.

---

## 1. Capa de salida y activación
* **Número de neuronas:** **$5$ neuronas** (una por cada categoría).
* **Función de activación:** **Softmax** (si las clases son mutuamente excluyentes).
  * *Razón:* Produce una distribución de probabilidad válida donde cada valor está en $[0, 1]$ y la suma total de las 5 salidas es exactamente $1.0$.

---

## 2. Redimensionamiento necesario
Antes del `fit`, **todas las imágenes deben ser redimensionadas de $224 \times 224 \times 3$ a $21 \times 28 \times 3$** (mediante OpenCV, tf.image o el preprocesamiento del generador) para acoplarse a la dimensión estricta de entrada del modelo.

---

## 3. Propuesta de mejora y síntoma
* **Mejora propuesta:** Implementar **Batch Normalization** después de cada convolución.
* **Síntoma en train/val:** El entrenamiento avanza de forma excesivamente lenta, oscilando con mesetas en las pérdidas (*vanishing/exploding gradients*), requiriendo un `learning rate` minúsculo para converger.
