# Pregunta 9 — CNN vs MLP vs RNN

## 1. Selección de arquitectura por escenario
1. **Datos tabulares (promedio, faltas, horas):** **MLP (Perceptrón Multicapa)**, porque procesa vectores de atributos tabulares sin relaciones espaciales ni temporales inherentes.
2. **Clasificación de fotos de semáforo:** **CNN (Red Neuronal Convolucional)**, ya que extrae patrones visuales espaciales locales con invarianza a traslaciones.
3. **Autocompletado de texto/código:** **RNN / Transformer**, debido a que modela secuencias temporales dependientes del contexto y orden de los tokens precedentes.

---

## 2. Si aplastas la foto del semáforo y usas un MLP grande
* **¿Puede aprender?** Sí, en un dataset pequeño o fijo puede sobreajustarse y memorizarlo.
* **¿Qué se sacrifica respecto a una CNN?**
  * **Explosión de parámetros:** Conectar cada píxel a neuronas densas eleva masivamente el cómputo.
  * **Pérdida de la estructura 2D:** Rompe la noción de vecindad entre píxeles adyacentes.
  * **Invarianza espacial:** Si el semáforo cambia levemente de posición, el MLP lo percibe como un patrón completamente nuevo y falla al generalizar.
