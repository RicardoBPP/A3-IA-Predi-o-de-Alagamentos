"""
Script para:

1) Ler série de chuva do CGE (Butantã)
2) Consumir o WFS de alagamentos do GeoSampa
3) Filtrar ocorrências do Butantã (quando houver coluna de subprefeitura)
4) Agregar ocorrências por dia
5) Unir chuva + ocorrências em um único CSV para treino de IA

Rodar a partir da raiz do projeto:

    python src/data/build_labels_geosampa.py

Pré-requisitos (no seu venv):

    pip install geopandas requests openpyxl
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd
import requests


# ============================================================================
# CONFIGURAÇÕES BÁSICAS / CAMINHOS
# ============================================================================

# Arquivo de configuração com URLs (você já criou):
DATASETS_TXT = Path("src/data/datasets.txt")

# Caminho da planilha de chuva do CGE Butantã (a mesma do modelo)
CHUVA_EXCEL = Path("data/processed/CGESP-PROCESSADOS/CGESP_BUTANTA_2020_2025.xlsx")

# Saída com chuva + rótulos de alagamento (para treino de IA)
OUTPUT_CSV = Path("data/processed/labels_chuva_alagamento_butanta.csv")

# Colunas candidatas para subprefeitura (vamos tentar achar uma delas)
SUBPREF_COL_CANDIDATAS = [
    "SUBPREF",
    "SUBPREFE",
    "SUBPREFEITURA",
    "NM_SUBPREF",
    "nm_subprefeitura",  # esta é a que apareceu no WFS
]
VALOR_BUTANTA = "BUTANTÃ"  # vamos usar contains "BUTANT" para pegar com/sem acento

# Se você descobrir o nome exato da coluna de data, pode colocar aqui.
# Se deixar None, o script tenta detectar automaticamente.
DATA_COL_PREFERIDA: str | None = None

# Nome esperado da camada no WFS (typeName).
# Pelo metadado "risco_ocorrencia_alagamento", o padrão GeoServer é:
#   workspace:layername  ->  geoportal:risco_ocorrencia_alagamento
LAYER_NAME = "geoportal:risco_ocorrencia_alagamento"


# ============================================================================
# LEITURA DO DATASETS.TXT
# ============================================================================

def load_dataset_urls(path: Path) -> dict:
    """
    Lê src/data/datasets.txt e devolve um dicionário com:
        GEO_URL, WFS_URL, WMS_URL, DIC_URL

    Formato esperado (linhas de exemplo):

        GEO_URL = https://geosampa.prefeitura.sp.gov.br/
        WFS_URL = http://wfs.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wfs
        WMS_URL = http://wms.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wms
        DIC_URL = https://metadados...

    Comentários depois de # são ignorados.
    """
    urls: dict[str, str] = {}
    if not path.exists():
        print(f"[WARN] Arquivo {path} não encontrado; usando valores padrão embutidos.")
        return urls

    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, val = line.split("=", 1)
        key = key.strip()
        # remove comentário no fim da linha, se houver
        val = val.split("#", 1)[0].strip()
        urls[key] = val

    print("[INFO] URLs lidas de datasets.txt:", urls)
    return urls


URLS = load_dataset_urls(DATASETS_TXT)

# WFS_URL é o que realmente precisamos para trazer os dados
WFS_URL = URLS.get(
    "WFS_URL",
    "http://wfs.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wfs",
)


# ============================================================================
# FUNÇÕES AUXILIARES – CHUVA
# ============================================================================

def carregar_chuva_cge(caminho_excel: Path) -> pd.DataFrame:
    """
    Lê o Excel do CGE (BUTANTA_2020_2025) e retorna uma série diária:
        data | chuva_mm
    """
    if not caminho_excel.exists():
        raise FileNotFoundError(f"Arquivo de chuva não encontrado: {caminho_excel}")

    df_raw = pd.read_excel(caminho_excel, sheet_name="BUTANTA_2020_2025")

    # Mapa de meses PT-BR -> número
    mes_map = {
        "Jan": 1, "Fev": 2, "Mar": 3, "Abr": 4, "Mai": 5, "Jun": 6,
        "Jul": 7, "Ago": 8, "Set": 9, "Out": 10, "Nov": 11, "Dez": 12
    }
    df_raw["MesNum"] = df_raw["Mes"].map(mes_map)

    # Colunas de dias (1..31)
    day_cols = [c for c in df_raw.columns if isinstance(c, int) and 1 <= c <= 31]

    # “Derreter” colunas de dia em linhas
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

    df_chuva = df_long[["data", "chuva_mm"]].copy()
    df_chuva["data"] = df_chuva["data"].dt.date  # apenas data (sem hora)

    return df_chuva


# ============================================================================
# FUNÇÕES AUXILIARES – ALAGAMENTOS (WFS)
# ============================================================================

def detectar_coluna_data(gdf: gpd.GeoDataFrame) -> str:
    """
    Tenta detectar automaticamente a coluna de data mais provável.
    Se DATA_COL_PREFERIDA estiver preenchida e existir, usa ela.
    """
    if DATA_COL_PREFERIDA and DATA_COL_PREFERIDA in gdf.columns:
        return DATA_COL_PREFERIDA

    candidatos = [c for c in gdf.columns if "DATA" in c.upper() or "DT_" in c.upper()]
    if not candidatos:
        raise ValueError(
            "Não foi possível encontrar uma coluna de DATA automaticamente. "
            "Veja as colunas do GeoDataFrame e ajuste DATA_COL_PREFERIDA."
        )

    print(f"[INFO] Colunas candidatas a data encontradas: {candidatos}")
    print(f"[INFO] Usando a primeira por padrão: {candidatos[0]}")
    return candidatos[0]


def detectar_coluna_subpref(gdf: gpd.GeoDataFrame) -> str | None:
    """
    Tenta encontrar uma coluna de subprefeitura entre as candidatas.
    Se não encontrar, retorna None (nesse caso não filtra por Butantã).
    """
    for col in SUBPREF_COL_CANDIDATAS:
        if col in gdf.columns:
            print(f"[INFO] Coluna de subprefeitura encontrada: {col}")
            return col
    print("[WARN] Nenhuma coluna de subprefeitura encontrada entre candidatas; "
          "não será feito filtro por Butantã automaticamente.")
    return None


def carregar_alagamentos_wfs() -> gpd.GeoDataFrame:
    """
    Consulta o WFS do GeoSampa e retorna um GeoDataFrame com as ocorrências de alagamento.
    """
    if not WFS_URL:
        raise ValueError("WFS_URL não definido. Verifique o datasets.txt.")

    params = {
        "service": "WFS",
        "version": "1.0.0",
        "request": "GetFeature",
        "typeName": LAYER_NAME,
        "outputFormat": "application/json",
    }

    print("[INFO] Chamando WFS...")
    resp = requests.get(WFS_URL, params=params, timeout=60)
    resp.raise_for_status()

    full_url = resp.url
    print(f"[INFO] URL final do WFS: {full_url}")

    gdf = gpd.read_file(full_url)
    print(f"[INFO] GeoDataFrame carregado com {len(gdf)} registros.")
    print("[INFO] Colunas disponíveis:", list(gdf.columns))

    return gdf


def preparar_eventos_por_dia(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Filtra (se possível) apenas Butantã e agrega ocorrências por dia.
    Retorna DataFrame: data | n_eventos.
    """
    # 1) Filtrar por subprefeitura Butantã, se houver coluna
    col_subpref = detectar_coluna_subpref(gdf)
    if col_subpref is not None:
        temp = gdf[col_subpref].astype(str).str.upper()
        mask_butanta = temp.str.contains("BUTANT", na=False)  # BUTANTA / BUTANTÃ
        gdf = gdf[mask_butanta].copy()
        print(
            f"[INFO] Filtrando para registros da região do Butantã "
            f"({col_subpref}): {len(gdf)} registros restantes."
        )
    else:
        print("[WARN] Sem coluna de subprefeitura; usando TODAS as ocorrências do município.")

    if gdf.empty:
        raise ValueError("Não há registros de alagamento após o filtro. Verifique o filtro de Butantã.")

    # 2) Detectar coluna de data
    col_data = detectar_coluna_data(gdf)

    # 3) Converter para datetime.date
    gdf["data"] = pd.to_datetime(gdf[col_data], errors="coerce").dt.date
    gdf = gdf.dropna(subset=["data"]).copy()

    # 4) Agregar número de eventos por dia
    eventos_dia = (
        gdf.groupby("data")
        .size()
        .reset_index(name="n_eventos")
        .sort_values("data")
        .reset_index(drop=True)
    )

    print(f"[INFO] Tabela de eventos por dia criada com {len(eventos_dia)} linhas.")
    return eventos_dia


def unir_chuva_e_eventos(df_chuva: pd.DataFrame, eventos_dia: pd.DataFrame) -> pd.DataFrame:
    """
    Junta chuva diária do CGE com número de ocorrências de alagamento no dia.
    Cria colunas auxiliares:
      - n_eventos (contagem)
      - flag_evento (0/1) -> houve pelo menos 1 ocorrência no dia.
    """
    df_join = df_chuva.merge(eventos_dia, on="data", how="left")
    df_join["n_eventos"] = df_join["n_eventos"].fillna(0).astype(int)
    df_join["flag_evento"] = (df_join["n_eventos"] > 0).astype(int)
    return df_join


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=== Passo 1: Carregando chuva do CGE Butantã ===")
    df_chuva = carregar_chuva_cge(CHUVA_EXCEL)
    print(f"[INFO] Série de chuva: {len(df_chuva)} dias.")

    print("\n=== Passo 2: Carregando alagamentos via WFS do GeoSampa ===")
    gdf_alag = carregar_alagamentos_wfs()

    print("\n=== Passo 3: Preparando eventos por dia (Butantã) ===")
    eventos_dia = preparar_eventos_por_dia(gdf_alag)

    print("\n=== Passo 4: Unindo chuva + eventos ===")
    df_labels = unir_chuva_e_eventos(df_chuva, eventos_dia)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_labels.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"[OK] Arquivo salvo em: {OUTPUT_CSV.resolve()}")
    print(df_labels.head())


if __name__ == "__main__":
    main()
