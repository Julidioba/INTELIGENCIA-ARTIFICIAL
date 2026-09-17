# Reporte EDA – Operación Dino Crash
**Analista:** Julian

---

## 1. Problema y dataset (Misión 1)

### P1 – Muerte en el siguiente frame
* **Y (Objetivo):** Binaria (`0` = Sobre vive / `1` = Muerte en el frame inmediato $t+1$).
* **X (mínimo 5 variables):**
  1. `dist_obstacle` (Numérica): Distancia horizontal en pixeles al obstáculo más cercano.
  2. `speed` (Numérica): Velocidad actual del juego.
  3. `obstacle_type` (Categoría): Tipo de obstáculo (`cactus_small`, `cactus_large`, `bird`, `none`).
  4. `jump` (Binaria): Indica si el dinosaurio está actualmente ejecutando un salto.
  5. `dino_y_pos` / `ducking` (Binaria/Numérica): Altura vertical o si el dino está agachado (para evaluar si esquiva aves).
* **Granularidad:** Un frame de una partida (medido aproximadamente cada 16 ms a 60 FPS).
* **Tamaño mínimo de dataset:** Mínimo ~100,000 frames de ~500 partidas variadas. Al ser una variable binaria muy desbalanceada (la muerte ocurre en < 1% de los frames total), se necesitan muchísimas observaciones para capturar suficientes eventos de clase positiva (`died=1`).
* **Riesgo si está mal definido:** Confundir "muerte en el frame actual" con "muerte en el siguiente frame", lo que genera un retardo temporal en la predicción donde la IA reacciona cuando el choque ya sucedió.

### P2 – Puntuación final alcanzada
* **Y (Objetivo):** Numérica continua (Entero representando el `score` final al morir).
* **X (mínimo 5 variables):**
  1. `avg_speed` (Numérica): Velocidad promedio durante toda la sesión.
  2. `total_jumps` (Numérica): Frecuencia total de saltos realizados.
  3. `obstacles_encountered` (Numérica): Conteo total de obstáculos superados.
  4. `player_reaction_time_avg` (Numérica): Promedio de milisegundos que tarda en saltar desde que aparece el obstáculo.
  5. `session_duration_ms` (Numérica): Tiempo total transcurrido de la sesión.
* **Granularidad:** Una partida completa (una fila por `session_id`).
* **Tamaño mínimo de dataset:** ~1,000 a 5,000 partidas completas jugadas por usuarios con distintos niveles de habilidad.
* **Riesgo si está mal definido:** Evaluar la partida en formato frame-por-frame en lugar de nivel agregativo; el modelo aprendería a predecir el score acumulado momento a momento (que es banal) en vez del rendimiento global de la partida.

### P3 – Tipo de obstáculo próximo
* **Y (Objetivo):** Categoría multiclase (`cactus_small`, `cactus_large`, `bird`, `none`).
* **X (mínimo 5 variables):**
  1. `current_score` (Numérica): Puntuación actual (afecta la probabilidad de que aparezcan aves).
  2. `speed` (Numérica): Velocidad actual del juego.
  3. `frames_since_last_obstacle` (Numérica): Tiempo transcurrido desde el último obstáculo evadido.
  4. `time_ms` (Numérica): Tiempo total de la sesión actual.
  5. `last_obstacle_type` (Categoría): Tipo de obstáculo previo generado.
* **Granularidad:** Un evento/frame previo a la generación de un obstáculo en el límite derecho de la pantalla.
* **Tamaño mínimo de dataset:** ~10,000 apariciones de obstáculos.
* **Riesgo si está mal definido:** Tratarlo como clasificación binaria ignorando la variedad de obstáculos, provocando que la IA no distinga si debe saltar o agacharse (ej. contra aves a media altura).

---

## 2. Diccionario y muestra (Misión 2)

* **Patrón en `died=1` (Frame 82):**
  En el frame 82 la velocidad es `6.8`, el tipo es `cactus_small`, la distancia es críticamente baja (`dist_obstacle = 12 px`) y la variable `jump = 0`. El patrón evidencia un choque inevitable por no haber mantenido la acción de salto en el momento del impacto.
* **¿`score` es buena variable para predecir muerte en el siguiente frame?**
  No. El `score` es monotónicamente creciente con respecto al tiempo transcurrido; indica qué tan lejos ha llegado la partida pero no si la posición actual o los reflejos inmediatos van a provocar un choque.
* **Variables críticas faltantes para P1:**
  1. `dino_y_pos` / `is_ducking`: Altura actual del personaje y si está agachado (imprescindible para aves).
  2. `obstacle_y_pos`: Altura del obstáculo.
  3. `time_since_jump`: Cuántos frames lleva en el aire para calcular física de gravedad y tiempo de caída.
* **¿`died` sirve tal como está definida para P1?**
  No sirve para entrenar directamente $t+1$ si describe la muerte en el frame presente ($t$). Se requiere crear un *target shift* derivado: $Y_{t} = died_{t+1}$ (donde el objetivo en el frame $t$ es el valor que toma `died` en el frame $t+1$).

---

## 3. Checklist EDA (Misión 3)

### Respuestas a 3 preguntas elegidas del checklist
1. **¿La clase objetivo está balanceada? (Pregunta 3):**
   * *Hipótesis:* Extremadamente desbalanceada. En una partida promedio de 1,000 frames, solo 1 frame contiene `died=1` (~0.1% de casos positivos).
2. **¿Hay fugas de información / leakage? (Pregunta 4):**
   * *Hipótesis:* Sí, si incluimos columnas como "distancia recorrida tras el impacto" o si usamos variables derivadas del estado posterior al choque.
3. **¿Datos i.i.d.? (Pregunta 8):**
   * *Hipótesis:* No son independientes ni idénticamente distribuidos a nivel de frame. Los frames dentro de una misma sesión guardan una estricta dependencia temporal y correlación secuencial.

### Explicación sobre i.i.d. y división de datos (Pregunta 8)
Mezclar frames de la misma partida en los conjuntos de *train* y *test* (mediante un `train_test_split` aleatorio por filas) provoca una fuga temporal masiva. El modelo memorizará los frames $t-1$ y $t+1$ presentes en entrenamiento para predecir el frame $t$ en prueba. Para evaluar la capacidad real de generalización, la división **debe realizarse por sesiones completas (`session_id`)**, asegurando que partidas enteras queden reservadas exclusivamente para test.

### Ejemplo concreto de Data Leakage en P1
Si se incluye la variable `score` final de la sesión o el valor de `time_ms` máximo de la partida dentro de los predictores del frame $t$, el modelo sabrá por adelantado cuántos segundos durará la partida en total, prediciendo con 100% de precisión artificial cuándo va a morir el dinosaurio sin aprender la física o la lógica del juego.

---

## 4. Interpretación de resúmenes (Misión 4)

1. **Cuantificación del desbalance en P1:**
   * Total de frames: ~12,000.
   * Total de eventos de muerte (`died=1`): 50.
   * Porcentaje de clase positiva: $\frac{50}{12000} \approx 0.41\%$.
2. **Implicación para las métricas:**
   * La métrica de **Accuracy es inútil**: Un modelo ingenuo que prediga siempre "Sobrevive" (`died=0`) obtendrá un $99.59\%$ de accuracy sin haber aprendido nada.
   * Se debe priorizar **Recall** (capturar las muertes) combinado con **F1-Score** o el área bajo la curva **PR-AUC (Precision-Recall AUC)**.
3. **Utilidad de `dist_obstacle` como predictor:**
   * Los datos indican que las muertes se concentran casi exclusivamente cuando `dist_obstacle < 20`. Esto demuestra que es la variable con mayor poder predictivo y separación de clases.
4. **Distribución de `score`:**
   * Presenta una **cola larga hacia la derecha** ($\text{Media} = 28$, $\text{Mediana} = 18$, $\text{Máx} = 120$). Para modelos lineales se requiere una transformación logarítmica ($\log(1 + score)$) para reducir la asimetría, o bien recurrir a modelos basados en árboles de decisión que son inmunes a este sesgo.

---

## 5. Elección de modelo (Misiones 5-6)

### Tabla de modelos por escenario

| Escenario | Tras tu EDA, ¿qué fila de la guía aplica? | Modelo que propondrías | 2 condiciones del dataset que deben cumplirse |
| :--- | :--- | :--- | :--- |
| **P1** | Y binaria muy desbalanceada | Random Forest / XGBoost con `scale_pos_weight` | 1. Dataset masivo con abundantes ejemplos de la clase minoria.<br>2. Inexistencia de data leakage temporal. |
| **P2** | Y numérica (`score` final) | Árbol Regresor / Gradient Boosting Regressor | 1. Filas agregadas a nivel de partida entera (`session_id`).<br>2. Representatividad suficiente de jugadores novatos y expertos. |
| **P3** | Y categórica multiclase | Regresión Logística Multinomial / Random Forest | 1. Balance adecuado de eventos por tipo de obstáculo.<br>2. Inclusión de variables de contexto previa al evento. |

### Contraejemplos y reglas fijas

1. **Escenario donde un Árbol Profundo parece buena idea pero EDA lo desaconseja:**
   * Si contamos con pocas partidas (ej. solo 5 partidas / ~1,000 frames) y variables continuas muy correlacionadas. Un árbol profundo generará sobreajuste (*overfitting*) memorizando el trayecto de esas pocas partidas en lugar de aprender una regla general.
2. **Escenario donde una Red Neuronal tiene sentido:**
   * Cuando en lugar de telemetría procesada (`dist_obstacle`, `speed`) ingresamos los **frames crudos en formato de matriz de píxeles (imágenes directas)** de la pantalla mediante una Red Neuronal Convolucional (CNN), habiendo recopilado cientos de miles de frames.
3. **Comparativa: Reglas Fijas vs. Modelo Aprendido en P1:**
   * *Regla Fija (`if dist_obstacle < X and jump == 0: muerte`):*
     * **Ventajas:** Cero costo computacional de entrenamiento, interpretabilidad total y ejecución instantánea.
     * **Límites:** Falla drásticamente cuando la velocidad aumenta (los umbrales dinámicos cambian) o ante obstáculos aéreos (aves) que requieren agacharse en lugar de saltar.
   * *Modelo Aprendido:*
     * **Ventajas:** Ajusta automáticamente los umbrales adaptándolos a la velocidad de la pantalla y combinaciones complejas de variables.
     * **Límites:** Requiere un proceso riguroso de recolección de datos y ajuste de hiperparámetros para evitar falsos positivos.

---

## Síntesis

Para resolver el problema P1 pediría un dataset tabulado a nivel de frame que contenga telemetría física espacial (`dist_obstacle`, `speed`, `dino_y_pos`, `obstacle_type`), con estructuración por `session_id` y con el target desplazado a $t+1$. Tras confirmar mediante el EDA el desbalance extremo del target y eliminar la fuga de datos temporal, propondría un modelo **XGBoost con ajuste de pesos de clase (`scale_pos_weight`)** optimizado sobre la métrica **F1-Score**.