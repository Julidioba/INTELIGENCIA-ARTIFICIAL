# Pregunta 3 — ¿Qué cambia si meto más capas?

## Enunciado
Arquitectura base:
* Entrada: $(21 \times 28 \times 3)$
* $\rightarrow \text{Conv2D}(32, 3 \times 3, \text{same}) + \text{LeakyReLU}$
* $\rightarrow \text{MaxPool}(2 \times 2)$
* $\rightarrow \text{Flatten} \rightarrow \text{Dense}(32) \rightarrow \text{Dense}(nClasses, \text{softmax})$

Variante con segundo bloque convolucional:
$$\dots \rightarrow \text{Conv2D}(32) \rightarrow \text{Pool} \rightarrow \text{Conv2D}(64, 3 \times 3, \text{same}) \rightarrow \text{Pool} \rightarrow \text{Flatten} \dots$$

---

## 1. Tamaño tras el primer MaxPool(2×2)
La entrada es $21 \times 28$. Con `Conv2D(same)` se mantiene en $21 \times 28$.
Al aplicar `MaxPool(2x2)` con stride 2:
* Alto: $\lfloor 21 / 2 \rfloor = 10$ (u $11$ según modo de padding/ceil).
* Ancho: $28 / 2 = 14$.

El tamaño aproximado es **$10 \times 14 \times 32$** (o **$11 \times 14 \times 32$**).

---

## 2. Significado de los 64 filtros de la segunda Conv2D
Detectan **patrones más compuestos y abstractos**.
* **Analogía:** La primera capa ve bordes simples o gradientes locales; la segunda capa combina esas líneas para reconocer esquinas, texturas, arcos o siluetas elementales debido a que su **campo receptivo efectivo** es más amplio.

---

## 3. Riesgo y mitigación
* **Falla típica:** **Sobreajuste severo (Overfitting)**. La pérdida de entrenamiento (*train loss*) continúa cayendo hacia cero mientras la pérdida de validación (*val loss*) se estanca y empieza a subir de forma divergente.
* **Medida prioritaria:** **Data Augmentation (Aumento de datos)**.
  * *Justificación:* Al tener un dataset reducido y un modelo con mayor capacidad de representación, generar variaciones sintéticas (rotaciones, zoom, traslaciones) expande la diversidad sin reducir la capacidad del extractor jerárquico.
