import streamlit as st
import pandas as pd
import joblib
from datetime import date


# Caso queira usar os mesmos rótulos diretamente aqui
RISK_LABELS = {
    0: "Sem risco",
    1: "Risco de Inundação Transitável",
    2: "Risco de Inundação Intransitável",
}


@st.cache_resource
def load_model():
    """Carrega o modelo Random Forest treinado (3 classes)."""
    pacote = joblib.load("models/modelo_risco_inundacao_3classes.joblib")
    model = pacote["model"]
    features = pacote["features"]
    labels = pacote["labels"]  # não obrigatório, mas está salvo
    return model, features, labels


def main():
    st.title("MVP – IA de Risco de Inundação no Butantã")
    st.write(
        "Este protótipo usa um modelo de classificação treinado com chuvas diárias "
        "do CGE Butantã, para estimar o risco de inundação em 3 níveis."
    )

    model, features, labels = load_model()

    st.subheader("Entrada manual de dados de chuva")

    data_ref = st.date_input("Data de referência", value=date.today())

    col1, col2 = st.columns(2)
    with col1:
        chuva_hoje = st.number_input("Chuva HOJE (mm)", min_value=0.0, step=0.1, value=0.0)
        chuva_lag1 = st.number_input("Chuva ONTEM (mm)", min_value=0.0, step=0.1, value=0.0)
    with col2:
        chuva_lag2 = st.number_input("Chuva 2 dias atrás (mm)", min_value=0.0, step=0.1, value=0.0)
        chuva_lag3 = st.number_input("Chuva 3 dias atrás (mm)", min_value=0.0, step=0.1, value=0.0)

    if st.button("Calcular risco"):
        mes = data_ref.month
        dia_semana = data_ref.weekday()  # 0 = segunda

        # Monta o dataframe com as MESMAS features usadas no treino
        X_novo = pd.DataFrame([{
            "chuva_mm": chuva_hoje,
            "chuva_lag1": chuva_lag1,
            "chuva_lag2": chuva_lag2,
            "chuva_lag3": chuva_lag3,
            "mes": mes,
            "dia_semana": dia_semana,
        }])

        # Garante a ordem correta das colunas
        X_novo = X_novo[features]

        # Predição
        probs = model.predict_proba(X_novo)[0]
        classe_pred = int(model.predict(X_novo)[0])

        st.subheader("Resultado")

        st.markdown(f"**Classe prevista:** {RISK_LABELS.get(classe_pred, 'Desconhecido')}")

        st.markdown("**Probabilidade por classe:**")
        for cls, p in zip(model.classes_, probs):
            st.write(f"- {RISK_LABELS[int(cls)]}: {p:.1%}")

        st.info(
            "Obs.: por enquanto, as classes de risco usam apenas faixas de chuva diária "
            "(Sem risco < 5 mm, Transitável entre 5 e 25 mm, Intransitável ≥ 25 mm). "
            "Quando tivermos o limiar físico (Pcrit) de escoamento para o ponto, "
            "vamos ajustar esses limites e re-treinar o modelo."
        )


if __name__ == "__main__":
    main()
