# Entrega2
# Proyecto de Aprendizaje Automático: Predicción de Reingreso Hospitalario en Pacientes Diabéticos

Trabajo realizado por **Alejandra Álvarez**, **Claudia de la Serna** y **Catalina Lara** para la asignatura de **Aprendizaje Automático** (Grado en Ingeniería Matemática, UFV, Curso 2026–2027).

---

## 📌 Descripción del Proyecto
Este proyecto tiene como objetivo desarrollar modelos de aprendizaje automático capaces de predecir si un paciente diabético reingresará en el hospital en un plazo inferior a 30 días (`readmitted == "<30"`), utilizando el dataset público *Diabetes 130-US Hospitals for Years 1999–2008*.

Anticipar los reingresos hospitalarios tempranos permite optimizar los recursos sanitarios y mejorar la atención preventiva de los pacientes con mayor riesgo clínico.

---

## 📂 Contenido del Repositorio

* **`entrega2_basico.py`**: Script principal que engloba:
  1. Carga del dataset y filtrado inicial (eliminación de fallecidos y cuidados paliativos).
  2. Análisis Exploratorio de Datos (EDA), cálculo de porcentajes de valores ausentes, análisis de correlación numérica y estudio del reingreso según hospitalizaciones previas.
  3. Preprocesamiento, codificación *one-hot* y partición robusta sin fuga de datos (`GroupShuffleSplit` por paciente).
  4. Entrenamiento y evaluación de dos modelos de clasificación: **Regresión Logística** (con ponderación de clases) y **Perceptrón Multicapa (MLP)** con submuestreo aleatorio (*undersampling*).
  5. Extracción de los coeficientes más influyentes de la Regresión Logística.
* **Gráficos generados**:
  * `grafico_readmitted.png`: Distribución de la variable objetivo tras el filtrado.
  * `grafico_correlaciones.png`: Correlación de las variables numéricas con el reingreso temprano.
  * `grafico_hospitalizaciones_previas.png`: Tasa de reingreso según el número de ingresos previos.

---

## 📊 Resumen Metodológico y Resultados Preliminares

1. **Filtrado y Limpieza:** Se descartaron los episodios de pacientes fallecidos o derivados a cuidados paliativos, dejando un total de **99.343 episodios** (de 69.990 pacientes distintos). La clase positiva (<30 días) representa el **11,39 %** del total.
2. **Factores Clave:** El análisis exploratorio demostró que el número de hospitalizaciones previas (`number_inpatient`) es la variable con mayor correlación y peso predictivo sobre el reingreso temprano (con tasas que escalan desde el 8,6 % sin ingresos previos hasta el 37,1 % con 5 o más ingresos).
3. **Modelado:** 
   * Se aplicó una partición agrupada por paciente (`80 % entrenamiento / 20 % test`) para evitar la fuga de información (*data leakage*).
   * Se evaluaron una **Regresión Logística** y un **Perceptrón Multicapa (MLP)** sobre el conjunto de test real (19.802 episodios desbalanceados), obteniendo una exactitud global del ~67 %, una sensibilidad superior al 51 % y una precisión preliminar de 0,173.

---

## 🚀 Cómo ejecutar el código

1. Asegúrate de tener instalado Python y las librerías necesarias (`numpy`, `pandas`, `matplotlib`, `scikit-learn`).
2. Coloca el archivo `diabetic_data.csv` en la misma carpeta que el script.
3. Ejecuta el script desde tu terminal:
   ```bash
   python entrega2_basico.py
