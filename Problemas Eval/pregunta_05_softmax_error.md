# Pregunta 5 — Softmax y el error “casi seguro”

## Enunciado
Clases: `[inundación, incendio, tornado]`
* Predicción ($P$): $[0.10, 0.05, 0.85]$
* Etiqueta real ($Y$): $[0, 1, 0]$ (incendio)

---

## 1. Clase que predice el modelo (`argmax`)
$$\text{argmax}(P) = \text{índice } 2 \rightarrow \mathbf{Tornado}$$
Predice **tornado** con una probabilidad del $85\%$.

---

## 2. ¿Está seguro o confundido?
El modelo está **muy seguro**, pero completamente equivocado.
Asignó el **$85\%$ de probabilidad a la clase errónea** (tornado) y apenas un **$5\%$ a la clase correcta** (incendio). No presenta confusión distributiva (como $0.34, 0.33, 0.33$), sino una falsa certeza.

---

## 3. Categorical Crossentropy: ¿Alta o baja?
La pérdida será **muy alta**.

**Justificación:**
La fórmula de entropía cruzada categórica para una muestra es:
$$\mathcal{L} = -\sum_{i} y_i \ln(p_i) = -\ln(p_{\text{correcta}})$$
Como la clase correcta es incendio ($p = 0.05$):
$$\mathcal{L} = -\ln(0.05) \approx 2.996$$
La función castiga logarítmicamente cuando la probabilidad asignada a la etiqueta real tiende a cero.
