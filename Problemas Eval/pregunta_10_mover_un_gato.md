# Pregunta 10 — Experimento mental: mover un gato

## Enunciado
Detección de gato en fotos de $64 \times 64$. En test, el gato aparece corrido 10 píxeles a la derecha.

---

## 1. ¿Por qué la CNN tiene más chance de acertar que un MLP?
Gracias a los **pesos compartidos** y la operación de convolución: el mismo kernel que aprendió a detectar orejas, ojos o pelaje escanea toda la imagen de izquierda a derecha. Al desplazarse el gato, las activaciones espaciales en el mapa de características simplemente se desplazan la misma cantidad de posiciones (**equivarianza por traslación**), la cual el subsiguiente pooling vuelve invariante.

---

## 2. Si el gato se gira 90° (sin haber visto gatos rotados)
**No garantiza acierto.** Las convoluciones estándar son invariantes/equivariantes a traslación, pero **no son intrínsecamente invariantes a la rotación**.
* **Solución necesaria:** Aplicar **Data Augmentation** incluyendo rotaciones aleatorias durante el entrenamiento o recolectar datos reales con orientaciones diversas.

---

## 3. Pregunta conceptual para un compañero
> *"Si en una capa Conv2D aumentamos la resolución de la imagen de entrada de $64 \times 64$ a $512 \times 512$, ¿aumenta el número de pesos que debemos entrenar en esa capa? Justifica por qué sí o por qué no."*
