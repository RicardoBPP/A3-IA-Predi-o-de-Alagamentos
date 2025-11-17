# 🛰️ Predição de Alagamentos Urbanos – Butantã/SP

Repositório do projeto de **IA para nowcasting e classificação de inundações urbanas** na região do **Butantã – São Paulo/SP**.  
A ideia é combinar **modelos hidrológicos simplificados**, **lógica fuzzy (FIS/ANFIS)** e **modelos de Machine Learning** (ex.: LightGBM / GRU) para gerar um **score de risco interpretável** e classes de impacto (Sem, Leve, Moderada, Severa).

---

## 🎯 Objetivo

Desenvolver um sistema que:

- **Integre dados de múltiplas fontes** (chuva, nível de rio, alagamentos reportados, etc.);
- Calcule **variáveis hidrológicas relevantes** (API, escoamento estimado, excedência de capacidade hidráulica);
- Gere **alertas em tempo quase real (nowcasting)** para pontos críticos no Butantã;
- Produza **saídas interpretáveis** (regras fuzzy + score + classes de risco).

---

## 🧱 Arquitetura (visão geral)

> *Protótipo – ajustar depois conforme o MVP for fechando.*

- **Camada de dados**
  - Coleta de dados brutos dos provedores:
    - CGE-SP – chuva / alertas
    - ANA / DAEE-SP – nível de rios
    - INMET – séries históricas de chuva
    - OpenWeather – previsão e dados recentes via API
  - Organização em `data/raw` e `data/metadata.xlsx`

- **Camada de pré-processamento**
  - Limpeza e padronização de datas, unidades e estações
  - Filtro para região do **Butantã**
  - Cálculo de variáveis derivadas (API, acumulados, intensidades etc.)

- **Camada de modelagem**
  - **FIS/ANFIS** para score de risco interpretável
  - Modelo de ML (ex.: LightGBM / GRU) para aprender o **resíduo** do FIS
  - Histerese e lógica de **mudança de classe** (evitar oscilação brusca de alertas)

- **Camada de saída**
  - Classes de risco: `0 = Sem`, `1 = Leve`, `2 = Moderada`, `3 = Severa`
  - Geração de mapas/pontos com alertas por área ou estação
  - Relatórios e gráficos para avaliação do modelo

---

## 📂 Estrutura do Repositório (proposta)

```bash
.
├── data/
│   ├── raw/              # dados brutos baixados dos provedores
│   ├── processed/        # dados tratados / features
│   └── metadata.xlsx     # dicionário de dados / estações / fontes
├── notebooks/
│   ├── 01_exploracao_dados.ipynb
│   ├── 02_tratamento_chuva.ipynb
│   └── 03_modelagem_fis_ml.ipynb
├── src/
│   ├── data/
│   │   ├── download_cgesp.py
│   │   ├── download_inmet.py
│   │   └── ...
│   ├── features/
│   │   ├── chuva_aggregations.py
│   │   └── hidrologia.py
│   ├── models/
│   │   ├── fis/
│   │   └── ml/
│   └── utils/
├── models/               # modelos treinados (salvos)
├── docs/                 # relatórios, diagramas, apresentações
├── requirements.txt
├── .env.example          # exemplo de variáveis de ambiente (chaves de API)
└── README.md
