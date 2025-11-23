import datetime as dt

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# CONFIGURAÇÕES BÁSICAS
# -----------------------------------------------------------------------------

MODEL_EVENTO_LR_PATH = "models/modelo_evento_butanta_lr.joblib"

st.set_page_config(
    page_title="Protótipo – Risco de Inundação (Butantã)",
    layout="centered",
)


# -----------------------------------------------------------------------------
# FUNÇÕES DE NEGÓCIO
# -----------------------------------------------------------------------------

def classificar_risco_chuva_label(mm_24h: float) -> str:
    """
    Regras de risco por chuva diária (mm), de acordo com os limites definidos:

      - Classe 0 – Sem risco: chuva < 20 mm
      - Classe 1 – Risco Transitável: 20 mm ≤ chuva < 60 mm
      - Classe 2 – Risco Intransitável: chuva ≥ 60 mm
    """
    if mm_24h < 20.0:
        return "Sem risco"
    elif mm_24h < 60.0:
        return "Risco de Inundação Transitável"
    else:
        return "Risco de Inundação Intransitável"


def add_lags_and_features_app(df: pd.DataFrame) -> pd.DataFrame:
    """
    Versão simplificada do feature engineering usado no treino,
    aplicada a um pequeno DataFrame com a sequência de dias.

    Espera colunas: data, chuva_mm.
    """

    df = df.sort_values("data").reset_index(drop=True)

    # Lags de chuva (dias anteriores)
    for k in [1, 2, 3]:
        df[f"chuva_lag{k}"] = df["chuva_mm"].shift(k)

    # Acumulados (3, 7, 30 dias)
    df["acum_3d"] = df["chuva_mm"].rolling(window=3, min_periods=1).sum()
    df["acum_7d"] = df["chuva_mm"].rolling(window=7, min_periods=1).sum()
    df["acum_30d"] = df["chuva_mm"].rolling(window=30, min_periods=1).sum()

    # API simples (índice de precipitação antecedente)
    k = 0.9
    api_vals = []
    prev = 0.0
    for v in df["chuva_mm"].fillna(0):
        prev = k * prev + v
        api_vals.append(prev)
    df["API"] = api_vals

    # Variáveis de calendário
    dt_series = pd.to_datetime(df["data"])
    df["ano"] = dt_series.dt.year
    df["mes"] = dt_series.dt.month
    df["dia"] = dt_series.dt.day
    df["dia_semana"] = dt_series.dt.weekday

    # Remove primeiras linhas sem lags completos
    df = df.dropna(subset=["chuva_lag1", "chuva_lag2", "chuva_lag3"]).reset_index(drop=True)

    return df


@st.cache_resource
def load_model_evento_lr():
    """
    Carrega o modelo de Regressão Logística treinado para
    ocorrência de alagamento no Butantã.
    """
    data = joblib.load(MODEL_EVENTO_LR_PATH)
    model = data["model"]
    features = data["features"]
    rain_min_mm = data["rain_min_mm"]
    return model, features, rain_min_mm


def calcular_prob_evento(
    chuva_hoje: float,
    chuva_ontem: float,
    chuva_2d: float,
    chuva_3d: float,
    data_ref: dt.date,
):
    """
    Constrói uma pequena série de 4 dias (t-3, t-2, t-1, t) com as chuvas informadas,
    aplica o mesmo tipo de feature engineering do treino e devolve
    o vetor de features para o dia t (data_ref).
    """

    # Monta a sequência de datas (t-3, t-2, t-1, t)
    datas = [
        data_ref - dt.timedelta(days=3),
        data_ref - dt.timedelta(days=2),
        data_ref - dt.timedelta(days=1),
        data_ref,
    ]
    chuvas = [chuva_3d, chuva_2d, chuva_ontem, chuva_hoje]

    df_seq = pd.DataFrame(
        {
            "data": datas,
            "chuva_mm": chuvas,
        }
    )

    df_feat = add_lags_and_features_app(df_seq)

    if df_feat.empty:
        return None  # não conseguiu formar lags suficientes

    # Pega a última linha (dia t = data_ref)
    row = df_feat.iloc[-1]
    return row


# -----------------------------------------------------------------------------
# INTERFACE STREAMLIT
# -----------------------------------------------------------------------------

st.title("🌧️ Protótipo – IA para Predição de Risco de Inundação")
st.subheader("Subprefeitura do Butantã – São Paulo/SP")

with st.expander("ℹ️ Sobre o Protótipo", expanded=False):
    st.write(
        """
        Este protótipo combina **regras simples por intensidade de chuva** com um
        **modelo de Regressão Logística** treinado a partir de dados históricos
        de chuva (CGE) e ocorrências de alagamento (GeoSampa) no Butantã.

        - A classificação **Sem risco / Transitável / Intransitável** é baseada em faixas de chuva em 24h.
        - O modelo binário estima a **probabilidade de ocorrência de alagamento em dias chuvosos**,
          considerando chuva atual, últimos dias e um índice simples de precipitação antecedente (API).
        """
    )

st.markdown("---")

st.header("1) Entradas de chuva")

col1, col2 = st.columns(2)
with col1:
    data_ref = st.date_input(
        "Data de referência (dia para o qual deseja estimar o risco)",
        value=dt.date.today(),
    )

with col2:
    st.write("")  # espaçamento
    st.write("Informe a chuva diária (mm) nos últimos dias:")

chuva_hoje = st.number_input("Chuva hoje (mm/24h)", min_value=0.0, step=1.0, value=0.0)
chuva_ontem = st.number_input("Chuva ontem (mm/24h)", min_value=0.0, step=1.0, value=0.0)
chuva_2d = st.number_input("Chuva há 2 dias (mm/24h)", min_value=0.0, step=1.0, value=0.0)
chuva_3d = st.number_input("Chuva há 3 dias (mm/24h)", min_value=0.0, step=1.0, value=0.0)

st.markdown("---")

if st.button("Calcular risco"):
    # ---------------------------------------------------------------------
    # 1) Risco por regra de chuva
    # ---------------------------------------------------------------------
    classe_regra = classificar_risco_chuva_label(chuva_hoje)
    st.subheader("Resultado 1 – Risco por intensidade de chuva (regras)")
    st.write(f"**Classe de risco (chuva em 24h): {classe_regra}**")

    # ---------------------------------------------------------------------
    # 2) Probabilidade de ocorrência de alagamento (modelo LR)
    #    Só faz sentido para dias com chuva acima do limiar de treino.
    # ---------------------------------------------------------------------
    try:
        modelo_lr, feature_cols, rain_min_mm = load_model_evento_lr()
    except Exception as e:
        st.error(f"Erro ao carregar modelo de ocorrência (LR): {e}")
    else:
        st.subheader("Resultado 2 – Probabilidade de ocorrência de alagamento")

        if chuva_hoje < rain_min_mm:
            st.info(
                f"A chuva informada ({chuva_hoje:.1f} mm) está abaixo do limiar "
                f"mínimo usado no treino do modelo ({rain_min_mm:.1f} mm). "
                "Neste caso, a probabilidade estimada de ocorrência é considerada muito baixa."
            )
            st.write("**Probabilidade estimada de ocorrência: ~0% (abaixo do limiar de chuva)**")
        else:
            # Monta features a partir da sequência de 4 dias
            row_feat = calcular_prob_evento(
                chuva_hoje=chuva_hoje,
                chuva_ontem=chuva_ontem,
                chuva_2d=chuva_2d,
                chuva_3d=chuva_3d,
                data_ref=data_ref,
            )

            if row_feat is None:
                st.warning(
                    "Não foi possível montar as features (lags) para o modelo. "
                    "Verifique se os valores de chuva fazem sentido."
                )
            else:
                # Garante que todas as features esperadas existem
                missing = [f for f in feature_cols if f not in row_feat.index]
                if missing:
                    st.error(f"Features ausentes para o modelo: {missing}")
                else:
                    X_input = row_feat[feature_cols].values.reshape(1, -1)
                    probs = modelo_lr.predict_proba(X_input)[0]
                    # Classe 1 = 'evento'
                    p_evento = float(probs[1])

                    st.write(f"**Probabilidade estimada de ocorrência de alagamento: {p_evento*100:.1f}%**")

                    # Interpretação qualitativa simples
                    if p_evento < 0.1:
                        nivel = "Baixa"
                    elif p_evento < 0.3:
                        nivel = "Moderada"
                    else:
                        nivel = "Alta"

                    st.write(f"Nível qualitativo de risco (modelo de ocorrência): **{nivel}**")

                    st.caption(
                        "Obs.: este modelo foi treinado com poucos dias de ocorrência real, "
                        "devendo ser interpretado como um protótipo exploratório, "
                        "com foco em complementar as regras de chuva."
                    )
