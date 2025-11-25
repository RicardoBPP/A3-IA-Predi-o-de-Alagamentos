"""
Treino de modelo binário para ocorrência de alagamento no Butantã.

Alvo: flag_evento (0 = sem ocorrência, 1 = houve pelo menos 1 ocorrência no dia)
Entradas: chuva do dia, lags, acumulados, API, variáveis de calendário.

Estratégia:
- Usamos apenas dias com chuva >= RAIN_MIN_MM (dias secos são tratados por regra).
- Fazemos oversampling manual da classe positiva (dias com evento) no CONJUNTO DE TREINO,
  para reduzir o desbalanceamento extremo.

Rodar a partir da raiz do projeto:

    python src/training/treino_modelo_evento_butanta.py

Pré-requisitos (no venv):
    pip install pandas numpy scikit-learn joblib
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib


# -----------------------------------------------------------------------------
# CAMINHOS
# -----------------------------------------------------------------------------

LABELS_CSV = Path("data/processed/labels_chuva_alagamento_butanta.csv")

# Agora vamos salvar DOIS modelos:
MODEL_PATH_RF = Path("models/modelo_evento_butanta_rf.joblib")
MODEL_PATH_LR = Path("models/modelo_evento_butanta_lr.joblib")

# Limiar mínimo de chuva para considerar dia "relevante"
# Abaixo disso, no uso do modelo você pode assumir "sem risco" por regra.
RAIN_MIN_MM = 5.0


# -----------------------------------------------------------------------------
# FUNÇÕES DE FEATURE ENGINEERING
# -----------------------------------------------------------------------------

def add_lags_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe DataFrame com colunas:
        data | chuva_mm | n_eventos | flag_evento

    Retorna DataFrame com features numéricas e alvo "flag_evento".
    """

    df = df.sort_values("data").reset_index(drop=True)

    # Lags de chuva (dias anteriores)
    for k in [1, 2, 3]:
        df[f"chuva_lag{k}"] = df["chuva_mm"].shift(k)

    # Acumulados (3, 7, 30 dias)
    df["acum_3d"] = df["chuva_mm"].rolling(window=3, min_periods=1).sum()
    df["acum_7d"] = df["chuva_mm"].rolling(window=7, min_periods=1).sum()
    df["acum_30d"] = df["chuva_mm"].rolling(window=30, min_periods=1).sum()

    # API simples (Índice de precipitação antecedente)
    k = 0.9
    api_vals = []
    prev = 0.0
    for v in df["chuva_mm"].fillna(0):
        prev = k * prev + v
        api_vals.append(prev)
    df["API"] = api_vals

    # Variáveis de calendário
    dt = pd.to_datetime(df["data"])
    df["ano"] = dt.dt.year
    df["mes"] = dt.dt.month
    df["dia"] = dt.dt.day
    df["dia_semana"] = dt.dt.weekday  # 0 = segunda

    # Depois de gerar lags, ainda podemos ter NaN nos primeiros dias:
    df = df.dropna(subset=["chuva_lag1", "chuva_lag2", "chuva_lag3"]).reset_index(drop=True)

    return df


def oversample_minority(X_train: np.ndarray, y_train, target_ratio: float = 0.2):
    """
    Oversampling manual da classe minoritária (1).

    target_ratio: fração desejada de positivos no conjunto de treino (ex: 0.2 -> 20%)

    Retorna:
        X_train_bal, y_train_bal
    """
    # Garante que y_train é um array NumPy (e não uma Series do pandas)
    y_arr = np.asarray(y_train)

    idx_pos = np.where(y_arr == 1)[0]
    idx_neg = np.where(y_arr == 0)[0]

    n_pos = len(idx_pos)
    n_neg = len(idx_neg)

    print(f"[INFO] Oversampling: antes - pos={n_pos}, neg={n_neg}")

    if n_pos == 0:
        print("[WARN] Não há exemplos positivos no treino; não é possível fazer oversampling.")
        return X_train, y_arr

    # Quantidade desejada de positivos para atingir aproximadamente target_ratio
    # target_ratio = n_pos_target / (n_pos_target + n_neg)  -> resolve pra n_pos_target
    n_pos_target = int((target_ratio * n_neg) / (1 - target_ratio))
    n_pos_target = max(n_pos_target, n_pos)   # pelo menos o que já temos

    # Limita para não exagerar demais (até 3x os negativos)
    n_pos_target = min(n_pos_target, 3 * n_neg)

    # Quantas cópias completas + resto
    reps = n_pos_target // n_pos
    rem = n_pos_target % n_pos

    idx_pos_rep = np.tile(idx_pos, reps)
    if rem > 0:
        extra = np.random.choice(idx_pos, size=rem, replace=True)
        idx_pos_rep = np.concatenate([idx_pos_rep, extra])

    # Constrói conjunto balanceado: todos negativos + positivos replicados
    X_neg = X_train[idx_neg]
    y_neg = y_arr[idx_neg]

    X_pos_bal = X_train[idx_pos_rep]
    y_pos_bal = y_arr[idx_pos_rep]

    X_bal = np.vstack([X_neg, X_pos_bal])
    y_bal = np.concatenate([y_neg, y_pos_bal])

    # Embaralha
    idx_perm = np.random.permutation(len(y_bal))
    X_bal = X_bal[idx_perm]
    y_bal = y_bal[idx_perm]

    print(f"[INFO] Oversampling: depois - pos={int((y_bal == 1).sum())}, "
          f"neg={int((y_bal == 0).sum())}")
    return X_bal, y_bal


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Arquivo de labels não encontrado: {LABELS_CSV}")

    print(f"[INFO] Lendo labels de: {LABELS_CSV}")
    df = pd.read_csv(LABELS_CSV, parse_dates=["data"])
    print(f"[INFO] Tamanho inicial: {df.shape}")

    # ------------------------------------------------------------------
    # 1) Filtrar apenas dias com chuva >= RAIN_MIN_MM
    #    (dias secos / quase secos serão tratados por regra, fora do modelo)
    # ------------------------------------------------------------------
    df_rain = df[df["chuva_mm"] >= RAIN_MIN_MM].copy()
    print(f"[INFO] Dias com chuva >= {RAIN_MIN_MM} mm: {df_rain.shape[0]}")

    eventos_pos = int((df_rain["flag_evento"] == 1).sum())
    eventos_neg = int((df_rain["flag_evento"] == 0).sum())
    print(f"[INFO] Dentro desse subset, dias com evento: {eventos_pos}, sem evento: {eventos_neg}")

    if eventos_pos == 0:
        print("[WARN] Não há nenhum dia com evento nesse subset de chuva. "
              "Verifique o RAIN_MIN_MM ou os dados.")
        return

    # Feature engineering nesse subconjunto
    df_feat = add_lags_and_features(df_rain)
    print(f"[INFO] Tamanho após feature engineering (chuva >= {RAIN_MIN_MM}): {df_feat.shape}")
    print("[INFO] Colunas disponíveis:", list(df_feat.columns))

    # Alvo: flag_evento (0/1)
    y = df_feat["flag_evento"].astype(int)

    # Features numéricas
    feature_cols = [
        "chuva_mm",
        "chuva_lag1", "chuva_lag2", "chuva_lag3",
        "acum_3d", "acum_7d", "acum_30d",
        "API",
        "mes", "dia_semana",
    ]
    X = df_feat[feature_cols].values

    print("[INFO] Número de dias com evento (1) no subset chuvoso:", int((y == 1).sum()))
    print("[INFO] Número de dias sem evento (0) no subset chuvoso:", int((y == 0).sum()))

    # Split treino/teste (ainda desbalanceado, por enquanto)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
            stratify=y,
        )
    except ValueError:
        # Se der erro (poucos positivos para stratify), faz split simples
        print("[WARN] Stratify falhou (poucos positivos). Fazendo split simples sem estratificar.")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
        )

    print("[INFO] Tamanho treino (antes oversampling):", X_train.shape, "teste:", X_test.shape)

    # ------------------------------------------------------------------
    # 2) Oversampling da classe positiva no TREINO
    # ------------------------------------------------------------------
    X_train_bal, y_train_bal = oversample_minority(X_train, y_train.values, target_ratio=0.2)
    print("[INFO] Tamanho treino (depois oversampling):", X_train_bal.shape)

    # ------------------------------------------------------------------
    # Modelo 1: Regressão Logística (binária)
    # ------------------------------------------------------------------
    log_reg = LogisticRegression(
        max_iter=1000,
        class_weight=None,  # já fizemos oversampling, não precisamos de class_weight
    )
    log_reg.fit(X_train_bal, y_train_bal)
    y_pred_lr = log_reg.predict(X_test)

    print("\n=== Regressão Logística (alarme de alagamento | chuva >= limiar, com oversampling) ===")
    print(confusion_matrix(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr, digits=3, zero_division=0))

    # ------------------------------------------------------------------
    # Modelo 2: Random Forest
    # ------------------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight=None,  # idem: usamos oversampling
    )
    rf.fit(X_train_bal, y_train_bal)
    y_pred_rf = rf.predict(X_test)

    print("\n=== Random Forest (alarme de alagamento | chuva >= limiar, com oversampling) ===")
    print(confusion_matrix(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf, digits=3, zero_division=0))

    # ------------------------------------------------------------------
    # Salvar modelos
    # ------------------------------------------------------------------
    MODEL_PATH_LR.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": log_reg,
            "features": feature_cols,
            "rain_min_mm": RAIN_MIN_MM,
        },
        MODEL_PATH_LR,
    )
    print(f"[OK] Modelo Regressão Logística salvo em: {MODEL_PATH_LR.resolve()}")

    joblib.dump(
        {
            "model": rf,
            "features": feature_cols,
            "rain_min_mm": RAIN_MIN_MM,
        },
        MODEL_PATH_RF,
    )
    print(f"[OK] Modelo Random Forest salvo em: {MODEL_PATH_RF.resolve()}")


if __name__ == "__main__":
    main()
