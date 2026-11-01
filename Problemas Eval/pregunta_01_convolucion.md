# Pregunta 1 — Convolución a mano (3×3)

## Enunciado
Tienes este trozo de imagen en escala de grises (un canal) y este kernel:

```text
Imagen 5×5                 Kernel 3×3
0  0  1  0  0              1  0 -1
0  1  2  1  0              1  0 -1
1  2  3  2  1              1  0 -1
0  1  2  1  0
0  0  1  0  0
```

---

## 1. Tamaño del mapa de salida
Con $stride = 1$ y $padding = valid$ (sin relleno):

$$\text{Salida} = \left\lfloor \frac{N - K}{S} \right\rfloor + 1 = \left\lfloor \frac{5 - 3}{1} \right\rfloor + 1 = 2 + 1 = 3$$

El mapa de salida tiene dimensiones **$3 \times 3$**.

---

## 2. Cálculo de la esquina superior izquierda
Submatriz $3 \times 3$ correspondiente:
$$\begin{pmatrix} 0 & 0 & 1 \\ 0 & 1 & 2 \\ 1 & 2 & 3 \end{pmatrix}$$

Kernel:
$$\begin{pmatrix} 1 & 0 & -1 \\ 1 & 0 & -1 \\ 1 & 0 & -1 \end{pmatrix}$$

Suma de productos elemento a elemento:
$$\text{Resultado} = (0 \cdot 1 + 0 \cdot 0 + 1 \cdot -1) + (0 \cdot 1 + 1 \cdot 0 + 2 \cdot -1) + (1 \cdot 1 + 2 \cdot 0 + 3 \cdot -1)$$
$$\text{Resultado} = (-1) + (-2) + (1 - 3) = -1 - 2 - 2 = -5$$

El valor en la posición $(0, 0)$ es **$-5$**.

---

## 3. Tipo de detector de bordes
El kernel detecta bordes **verticales**.

**Justificación:**
La columna izquierda tiene coeficientes positivos ($+1$), la columna central tiene ceros ($0$) y la columna derecha tiene coeficientes negativos ($-1$). Las transiciones bruscas de intensidad se miden en el eje horizontal ($X$), respondiendo ante líneas o límites verticales de contraste.
