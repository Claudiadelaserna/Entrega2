import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 1. Cargar y quitar fallecidos/paliativos (no pueden reingresar)
df = pd.read_csv("diabetic_data.csv", na_values="?", low_memory=False)
df = df[~df["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21])]
df["reingreso_30"] = (df["readmitted"] == "<30").astype(int)

# 2. Datos básicos
print("Episodios:", len(df))
print("Pacientes distintos:", df["patient_nbr"].nunique())
print("Reingresos <30 días:", round(df["reingreso_30"].mean() * 100, 2), "%")

# 3. Valores ausentes (solo los 5 peores)
print("\nMás valores ausentes (%):")
print((df.isna().mean() * 100).round(1).sort_values(ascending=False).head(5))

# 4. Correlación de las variables numéricas con el reingreso
numericas = ["time_in_hospital", "num_lab_procedures", "num_procedures",
             "num_medications", "number_outpatient", "number_emergency",
             "number_inpatient", "number_diagnoses"]
corr = df[numericas + ["reingreso_30"]].corr()["reingreso_30"].drop("reingreso_30")
corr = corr.sort_values(ascending=False)
print("\nCorrelación con el reingreso:")
print(corr.round(3))

# 5. Tasa de reingreso según hospitalizaciones previas
grupo = df["number_inpatient"].clip(upper=5)
tasa = (df.groupby(grupo)["reingreso_30"].mean() * 100).round(1)
print("\n% de reingreso según hospitalizaciones previas (5 = 5 o más):")
print(tasa)

# 6. Dos gráficos
plt.figure(figsize=(6, 4))
plt.bar(tasa.index.astype(str), tasa.values)
plt.xlabel("Hospitalizaciones previas (5 = 5 o más)")
plt.ylabel("% de reingresos <30 días")
plt.title("Reingreso según hospitalizaciones previas")
plt.tight_layout()
plt.savefig("grafico_hospitalizaciones_previas.png")
plt.close()

plt.figure(figsize=(7, 4))
plt.barh(corr.index[::-1], corr.values[::-1])
plt.xlabel("Correlación con el reingreso <30 días")
plt.title("Variables numéricas y reingreso")
plt.tight_layout()
plt.savefig("grafico_correlaciones.png")
plt.close()

# 7. EDA (análisis exploratorio)
print("\nReingresos (readmitted):")
print(df["readmitted"].value_counts())
print((df["readmitted"].value_counts(normalize=True) * 100).round(2))

print("\nPorcentaje de valores ausentes (solo columnas con ausentes):")
ausentes = (df.isna().mean() * 100).round(2)
print(ausentes[ausentes > 0].sort_values(ascending=False))

print("\nPacientes distintos:", df["patient_nbr"].nunique())

df["readmitted"].value_counts().plot(kind="bar")
plt.title("Reingresos")
plt.ylabel("Número de episodios")
plt.tight_layout()
plt.savefig("grafico_readmitted.png")
plt.close()

# 8. PREPROCESADO
df = df[~df["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21])]
print("\nFilas tras quitar fallecidos/paliativos:", len(df))

y = (df["readmitted"] == "<30").astype(int)
print("Porcentaje de clase positiva:", round(y.mean() * 100, 2), "%")

grupos = df["patient_nbr"]

X = df.drop(columns=["readmitted", "encounter_id", "reingreso_30", "patient_nbr",
                     "weight", "payer_code", "medical_specialty",
                     "diag_1", "diag_2", "diag_3"])

X = pd.get_dummies(X, dtype=int)
print("Número de variables tras la codificación:", X.shape[1])

# 9. DIVISIÓN ENTRENAMIENTO / TEST
division = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
idx_train, idx_test = next(division.split(X, y, groups=grupos))
X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
print("Entrenamiento:", len(X_train), "| Test:", len(X_test))

# 10. FUNCIÓN PARA MOSTRAR MÉTRICAS
def mostrar_metricas(nombre, y_real, y_pred, y_prob):
    print("\n---", nombre, "---")
    print("Exactitud (accuracy):", round(accuracy_score(y_real, y_pred), 3))
    print("Precisión (precision):", round(precision_score(y_real, y_pred), 3))
    print("Sensibilidad (recall):", round(recall_score(y_real, y_pred), 3))
    print("F1:", round(f1_score(y_real, y_pred), 3))
    print("AUC-ROC:", round(roc_auc_score(y_real, y_prob), 3))

# 11. MODELO 1: REGRESIÓN LOGÍSTICA
escalador = StandardScaler()
X_train_esc = escalador.fit_transform(X_train)
X_test_esc = escalador.transform(X_test)

logistica = LogisticRegression(class_weight="balanced", max_iter=1000)
logistica.fit(X_train_esc, y_train)
pred = logistica.predict(X_test_esc)
prob = logistica.predict_proba(X_test_esc)[:, 1]
mostrar_metricas("Regresión logística", y_test, pred, prob)

# 12. MODELO 2: PERCEPTRÓN MULTICAPA
positivos = (y_train == 1).values
X_pos = X_train_esc[positivos]
X_neg = X_train_esc[~positivos]
azar = np.random.RandomState(42)
elegidos = azar.choice(len(X_neg), size=len(X_pos), replace=False)
X_bal = np.vstack([X_pos, X_neg[elegidos]])
y_bal = np.array([1] * len(X_pos) + [0] * len(X_pos))
print("\nEntrenamiento equilibrado para el perceptrón:", len(y_bal), "episodios")

perceptron = MLPClassifier(hidden_layer_sizes=(32,), activation="logistic",
                           max_iter=200, early_stopping=True,
                           random_state=42)
perceptron.fit(X_bal, y_bal)
pred = perceptron.predict(X_test_esc)
prob = perceptron.predict_proba(X_test_esc)[:, 1]
mostrar_metricas("Perceptrón multicapa", y_test, pred, prob)

# 13. VARIABLES MÁS INFLUYENTES SEGÚN LA REGRESIÓN LOGÍSTICA
coeficientes = pd.Series(logistica.coef_[0], index=X.columns)
orden = coeficientes.abs().sort_values(ascending=False).head(10).index
print("\nTop 10 variables con mayor |coeficiente| en la regresión logística:")
print(coeficientes[orden].round(3))