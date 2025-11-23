# A3 – IA para Predição de Risco de Inundação no Butantã 🌧️🤖

Protótipo de sistema de IA para apoiar a **avaliação de risco de inundação** na Subprefeitura do **Butantã (São Paulo/SP)**, usando **dados públicos** de chuva (CGE-SP) e **ocorrências reais de alagamento** (GeoSampa).

O projeto foi desenvolvido como parte da disciplina **Sistemas de Controle e IA**, com foco em:

- construir um **Protótipo funcional**;
- documentar o pipeline completo de dados → modelos → interface;
- mostrar, na prática, as **limitações reais** de prever alagamentos só com dados abertos.

---

## 📌 Visão geral

O repositório implementa **dois modelos complementares**:

1. **Modelo 1 – 3 classes de risco (chuva ⇒ severidade)**  
   Classifica o risco com base na intensidade de chuva diária (mm/24h) no CGE Butantã:

   - Classe 0 – **Sem risco**: chuva \< 20 mm  
   - Classe 1 – **Risco de Inundação Transitável**: 20 mm ≤ chuva \< 60 mm  
   - Classe 2 – **Risco de Inundação Intransitável**: chuva ≥ 60 mm  

   👉 Entradas: chuva de hoje, lags (1, 2, 3 dias) e calendário (mês, dia da semana).  
   👉 Saída: classe de risco (0, 1, 2) + probabilidades.  
   👉 Front-end: app Streamlit `app/app_risco_inundacao.py`.

2. **Modelo 2 – Alarme binário em dias chuvosos (chuva ⇒ alagamento sim/não)**  
   Usa **chuva do CGE** + **ocorrências de alagamento do GeoSampa** (camada `risco_ocorrencia_alagamento`) filtradas para a **Subprefeitura do Butantã**.

   - Condiciona o problema a **dias com chuva ≥ 5 mm**  
   - Cria `flag_evento`: 0 = sem alagamento, 1 = com alagamento  
   - Features: lags, acumulados (3, 7, 30 dias), API (índice de precipitação antecedente) e calendário  
   - Lida com o **desbalanceamento extremo** (quase não há dias com evento) usando **oversampling** no treino  
   - Modelo final salvo em `models/modelo_evento_butanta.joblib`

> ⚠️ Importante: o modelo binário tem caráter de **prova de conceito**. Com tão poucos dias com alagamento, ele não é adequado como previsor operacional “de verdade”, mas é ótimo para mostrar as dificuldades do problema.

