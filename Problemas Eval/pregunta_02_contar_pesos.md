# Pregunta 2 — Contar pesos

## Enunciado
Imagen RGB de $28 \times 28 \times 3$.
* **Modelo A (MLP ingenuo):** Aplana la imagen a un vector y conecta todas las entradas a una capa densa de 64 neuronas (sin bias).
* **Modelo B (CNN):** Una `Conv2D` con 16 filtros de $3 \times 3$, `padding='same'`, sobre la misma imagen (con bias: 1 bias por filtro).

---

## 1. Pesos de la capa densa (Modelo A)
* Entradas: $28 \times 28 \times 3 = 2{,}352$
* Neuronas de salida: $64$
* Bias: $0$

$$\text{Pesos} = n_{in} \times n_{out} = 2{,}352 \times 64 = 150{,}528$$

Tiene **$150{,}528$ pesos**.

---

## 2. Parámetros de la Conv2D (Modelo B)
* Dimensiones del kernel: $3 \times 3$
* Canales de entrada ($C_{in}$): $3$
* Filtros de salida ($C_{out}$): $16$
* Biases: $1$ por filtro ($16$ en total)

$$\text{Parámetros} = (K_h \times K_w \times C_{in} \times C_{out}) + C_{out}$$
$$\text{Parámetros} = (3 \times 3 \times 3 \times 16) + 16 = (27 \times 16) + 16 = 432 + 16 = 448$$

Tiene **$448$ parámetros**.

---

## 3. ¿Por qué la CNN suele generalizar mejor?
La CNN generaliza mejor gracias al uso de **pesos compartidos y conectividad local**, lo que reduce drásticamente los parámetros libres evitando el sobreajuste y dotando al modelo de **invarianza frente a traslaciones espaciales**.
