import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


# Mapeamento numérico -> rótulo de texto
RISK_LABELS = {
    0: "Sem risco",
    1: "Risco de Inundação Transitável",
    2: "Risco de Inundação Intransitável",
}


def carregar_chuva_cge(caminho_excel: str) -> pd.DataFrame:
    """
    Lê o Excel do CGE (BUTANTA_2020_2025) e devolve uma tabela diária:
    data | chuva_mm
    """
    file_path = Path(caminho_excel)
    df_raw = pd.read_excel(file_path, sheet_name="BUTANTA_2020_2025")

    # Mapa de meses PT-BR -> número
    mes_map = {
        "Jan": 1, "Fev": 2, "Mar": 3, "Abr": 4, "Mai": 5, "Jun": 6,
        "Jul": 7, "Ago": 8, "Set": 9, "Out": 10, "Nov": 11, "Dez": 12
    }
    df_raw["MesNum"] = df_raw["Mes"].map(mes_map)

    # Colunas de dias (1..31)
    day_cols = [c for c in df_raw.columns
                if isinstance(c, int) and 1 <= c <= 31]

    # “Derreter” as colunas de dias em linhas
    df_long = df_raw.melt(
        id_vars=["Ano", "Mes", "MesNum", "Posto"],
        value_vars=day_cols,
        var_name="Dia",
        value_name="chuva_mm"
    )

    df_long["Dia"] = df_long["Dia"].astype(int)

    # Construir a data
    df_long["data"] = pd.to_datetime(
        dict(year=df_long["Ano"], month=df_long["MesNum"], day=df_long["Dia"]),
        errors="coerce"
    )

    # Remover datas inválidas e dias sem valor
    df_long = df_long.dropna(subset=["data", "chuva_mm"]).reset_index(drop=True)
    df_long = df_long.sort_values("data").reset_index(drop=True)

    return df_long


def classificar_risco_chuva(mm: float) -> int:
    """
    Define a classe de risco com base na chuva do dia (mm).

    0 -> Sem risco
    1 -> Risco de Inundação Transitável
    2 -> Risco de Inundação Intransitável
    """
    if mm < 20.0:
        return 0
    elif mm < 60.0:
        return 1
    else:
        return 2


def montar_dataset_ml(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features simples (lags + calendário) e a classe de risco (3 classes).
    Saída: df_ml com colunas de features + 'classe_risco'.
    """
    df_long = df_long.copy()

    # Classe de risco numérica
    df_long["classe_risco"] = df_long["chuva_mm"].apply(classificar_risco_chuva)

    df = df_long.set_index("data").sort_index()

    # Lags de chuva (1, 2, 3 dias atrás)
    df["chuva_lag1"] = df["chuva_mm"].shift(1)
    df["chuva_lag2"] = df["chuva_mm"].shift(2)
    df["chuva_lag3"] = df["chuva_mm"].shift(3)

    # Calendário
    df["mes"] = df.index.month
    df["dia_semana"] = df.index.dayofweek  # 0 = segunda

    # Remove as primeiras linhas com NaN de lag
    df_ml = df.dropna(subset=["chuva_lag1", "chuva_lag2", "chuva_lag3"]).copy()
    return df_ml


def treinar_modelos(df_ml: pd.DataFrame):
    """
    Separa treino/teste no tempo, treina Logística e Random Forest
    para 3 classes de risco, e mostra as métricas.
    """
    df_ml = df_ml.reset_index()  # garante coluna "data"
    corte = pd.Timestamp("2024-01-01")

    treino = df_ml[df_ml["data"] < corte].copy()
    teste = df_ml[df_ml["data"] >= corte].copy()

    features = [
        "chuva_mm",
        "chuva_lag1", "chuva_lag2", "chuva_lag3",
        "mes", "dia_semana",
    ]

    X_train = treino[features]
    y_train = treino["classe_risco"]
    X_test = teste[features]
    y_test = teste["classe_risco"]

    print(f"Tamanho treino: {X_train.shape}, teste: {X_test.shape}")

    # Modelo 1: Regressão Logística multiclasses
    log_reg = LogisticRegression(
        max_iter=1000,
        multi_class="auto",
        class_weight="balanced"
    )
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)

    print("\n=== Regressão Logística (3 classes) ===")
    print(confusion_matrix(y_test, y_pred_lr))
    print(classification_report(
        y_test,
        y_pred_lr,
        digits=3,
        target_names=[
            RISK_LABELS[0],
            RISK_LABELS[1],
            RISK_LABELS[2],
        ]
    ))

    # Modelo 2: Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced",  # classes desbalanceadas
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    print("\n=== Random Forest (3 classes) ===")
    print(confusion_matrix(y_test, y_pred_rf))
    print(classification_report(
        y_test,
        y_pred_rf,
        digits=3,
        target_names=[
            RISK_LABELS[0],
            RISK_LABELS[1],
            RISK_LABELS[2],
        ]
    ))

    return log_reg, rf, features


def main():
    # caminho correto do Excel a partir da raiz do projeto
    caminho_excel = "data/processed/CGESP-PROCESSADOS/CGESP_BUTANTA_2020_2025.xlsx"

    print("Lendo dados de chuva do CGE (Butantã)...")
    df_long = carregar_chuva_cge(caminho_excel)

    print("Montando dataset de ML (lags + calendário + classe de risco)...")
    df_ml = montar_dataset_ml(df_long)
    print("Dataset para ML:", df_ml.shape)

    print("Treinando modelos...")
    log_reg, rf, features = treinar_modelos(df_ml)

    # Salva o Random Forest e a lista de features em models/
    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": rf, "features": features, "labels": RISK_LABELS},
        "models/modelo_risco_inundacao_3classes.joblib"
    )
    print("\nModelo Random Forest salvo em 'models/modelo_risco_inundacao_3classes.joblib'.")


if __name__ == "__main__":
    main()
