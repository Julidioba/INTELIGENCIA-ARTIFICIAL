# Pregunta 7 — Dropout: ¿dónde y para qué?

## Enunciado
Capa densa de 32 neuronas con `Dropout(0.5)`.

---

## 1. Dropout(0.5) durante el entrenamiento
En cada paso de propagación hacia adelante (*forward pass*), apaga aleatoriamente el **$50\%$ de las neuronas** (sus salidas se fijan en 0).
* En promedio sobreviven **$16$ neuronas** por paso ($32 \times 0.5$).

---

## 2. Dropout durante inferencia (`predict`)
**No se apagan neuronas al azar.**
* Todas las neuronas permanecen activas ($100\%$) para aprovechar toda la capacidad del ensamble aprendido.
* Las activaciones o pesos se escalan adecuadamente (o se usa *inverted dropout* en entrenamiento dividiendo por $1 - p$) para igualar las magnitudes esperadas.

---

## 3. Crítica a usar Dropout solo en la última época
Esta propuesta **carece de sentido técnico**.
* El propósito de Dropout es prevenir la coadaptación de neuronas a lo largo de todo el proceso de optimización, forzando a la red a no depender de rutas fijas.
* Si se activa de golpe únicamente en la última época, desestabiliza las activaciones aprendidas y degrada la precisión inmediatamente sin aportar regularización previa.
