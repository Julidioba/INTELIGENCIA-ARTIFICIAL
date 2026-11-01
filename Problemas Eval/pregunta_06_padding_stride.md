# Pregunta 6 — Padding, stride y “se me achicó la imagen”

## Enunciado
Entrada $32 \times 32$, kernel $5 \times 5$, 1 canal, 1 filtro.

Fórmula:
$$\text{Salida} = \left\lfloor \frac{N - K + 2P}{S} \right\rfloor + 1$$

---

## 1. Tabla de salidas

| padding | stride | salida (alto×ancho) | Justificación matemática |
| :--- | :---: | :---: | :--- |
| **valid** | 1 | **$28 \times 28$** | $\lfloor(32 - 5)/1\rfloor + 1 = 27 + 1 = 28$ |
| **valid** | 2 | **$14 \times 14$** | $\lfloor(32 - 5)/2\rfloor + 1 = \lfloor 13.5 \rfloor + 1 = 14$ |
| **same**  | 1 | **$32 \times 32$** | Conserva el tamaño original mediante relleno de ceros |

---

## 2. Problema práctico de encadenar tres MaxPool(2×2)
* $32 \rightarrow 16 \rightarrow 8 \rightarrow 4$ píxeles.
* **Problema:** Se pierde casi la totalidad de la **resolución espacial**. Si el objeto a clasificar es pequeño (por ejemplo, un detalle de $3 \times 3$ píxeles), queda comprimido o totalmente borrado tras sucesivos submuestreos, perdiendo información discriminativa crítica.
