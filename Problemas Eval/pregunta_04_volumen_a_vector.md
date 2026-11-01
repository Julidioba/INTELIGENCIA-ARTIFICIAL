# Pregunta 4 — Formas: de volumen a vector

## Enunciado
Tensor antes de `Flatten`: forma `(7, 7, 64)`.

---

## 1. Número de valores que salen del Flatten
$$\text{Total} = 7 \times 7 \times 64 = 49 \times 64 = 3{,}136$$

Salen **$3{,}136$ números**.

---

## 2. Parámetros de la siguiente capa Dense(32) con bias
* Entradas ($n_{in}$): $3{,}136$
* Neuronas ($n_{out}$): $32$
* Biases: $32$

$$\text{Parámetros} = (n_{in} \times n_{out}) + n_{out} = (3{,}136 \times 32) + 32 = 100{,}352 + 32 = 100{,}384$$

Tiene **$100{,}384$ parámetros**.

---

## 3. Global Average Pooling (GAP)
* **Forma de salida:** Vector de **$64$ valores** (forma `(64,)`). Promedia cada mapa espacial $7 \times 7$ a un único escalar por canal.
* **Ventajas frente a Flatten:**
  1. Reduce drásticamente la cantidad de parámetros de la red, mitigando el sobreajuste.
  2. Permite que la red acepte imágenes de resolución variable sin romper la dimensionalidad fija que exige `Flatten` hacia la capa densa.
