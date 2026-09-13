from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ============================================================
# CONFIGURAÇÃO
# ============================================================
st.set_page_config(
    page_title="Violência contra a Mulher no Paraná — 2025",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dados"
ASSETS_DIR = ROOT / "assets"
BASE_PATH = DATA_DIR / "base_municipal_2025_com_ibge.csv"
FEM_SENASP_PATH = DATA_DIR / "feminicidio_senasp_2025.csv"
ESTUPRO_PATH = DATA_DIR / "estupro_senasp_2025.csv"
LESFEM_PATH = DATA_DIR / "lesfem_2025.csv"
LOCAL_GEOJSON_PATH = ASSETS_DIR / "pr_municipios.geojson"

# Paleta de cores
LAVENDER = "#C4ABEC"
PURPLE = "#A866BE"
MAGENTA = "#811F82"
SAGE = "#9EB990"
GREEN = "#344E2A"
PALETTE = [LAVENDER, PURPLE, MAGENTA, SAGE, GREEN]

# Garante que TODOS os gráficos Plotly Express que usam categorias
# adotem a paleta do projeto, em vez das cores padrão azul/vermelho do Plotly.
px.defaults.color_discrete_sequence = PALETTE

BG = "#09080C"
BG_2 = "#111015"
CARD = "#16131B"
CARD_2 = "#1B1721"
TEXT = "#F7F2FA"
MUTED = "#B9AFBF"
GRID = "rgba(196,171,236,.12)"
BORDER = "rgba(196,171,236,.20)"

SOURCES = {
    "IPARDES": "https://www.ipardes.pr.gov.br/Pagina/Base-de-Dados-do-Estado",
    "SENASP": "https://www.gov.br/mj/pt-br/assuntos/sua-seguranca/seguranca-publica/estatistica/dados-nacionais-1/base-de-dados-e-notas-metodologicas-dos-gestores-estaduais-sinesp-vde-2022-e-2023",
    "LESFEM": "https://sites.uel.br/lesfem/base-de-dados-e-nota-metodologica/",
    "CAPE": "https://www.seguranca.pr.gov.br/CAPE",
    "IBGE_MALHAS": "https://www.ibge.gov.br/geociencias/organizacao-do-territorio/malhas-territoriais/15774-malhas.html",
    "IBGE_CIDADES": "https://cidades.ibge.gov.br/",
}

# População usada como referência nos cálculos proporcionais.
# O painel adota a população do último Censo Demográfico disponível (IBGE, 2022)
# como denominador comum para as taxas municipais. A referência temporal é sempre
# exibida no painel para evitar confusão com uma população de 2025.
POPULATION_SOURCE_COL = "IBGE - População no último censo - pessoas [2022]"
POPULATION_COL = "População de referência"
POPULATION_LABEL = "População IBGE — Censo 2022"

# O mapa tenta primeiro a API oficial do IBGE. GitHub/CDN ficam apenas como fallback.
IBGE_GEO_URLS = [
    "https://servicodados.ibge.gov.br/api/v4/malhas/estados/41",
    "https://servicodados.ibge.gov.br/api/v3/malhas/estados/41",
]
IBGE_LOCALIDADES_URL = "https://servicodados.ibge.gov.br/api/v1/localidades/estados/41/municipios"
GEOJSON_FALLBACK_URLS = [
    "https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-41-mun.json",
    "https://cdn.jsdelivr.net/gh/tbrugz/geodata-br@master/geojson/geojs-41-mun.json",
]

# ============================================================
# ESTILO — DARK / MOBILE FIRST
# ============================================================
st.markdown(
    f"""
    <style>
        :root {{
            --bg: {BG};
            --bg2: {BG_2};
            --card: {CARD};
            --card2: {CARD_2};
            --text: {TEXT};
            --muted: {MUTED};
            --lavender: {LAVENDER};
            --purple: {PURPLE};
            --magenta: {MAGENTA};
            --sage: {SAGE};
            --green: {GREEN};
            --border: {BORDER};
        }}

        html, body, [class*="css"] {{
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}

        .stApp {{
            background:
                radial-gradient(circle at 84% 8%, rgba(129,31,130,.16), transparent 24rem),
                radial-gradient(circle at 4% 36%, rgba(52,78,42,.12), transparent 28rem),
                {BG};
            color: {TEXT};
        }}
        .app-caption {{
        color: #FFFFFF !important;
        font-size: 0.875rem;
        line-height: 1.55;
        margin: 0.15rem 0 0.8rem 0;
        }}

        .app-caption * {{
            color: #FFFFFF !important;
        }}

        .block-container {{
            max-width: 1500px;
            padding-top: 1.15rem;
            padding-bottom: 3.2rem;
        }}

        h1, h2, h3, h4 {{
            color: {TEXT};
            letter-spacing: -0.025em;
        }}

        p, li {{ line-height: 1.62; }}

        .hero {{
            border: 1px solid {BORDER};
            border-radius: 24px;
            padding: clamp(1.15rem, 3vw, 2.25rem);
            background:
                linear-gradient(135deg, rgba(129,31,130,.18), rgba(52,78,42,.08) 62%, rgba(196,171,236,.05)),
                rgba(17,16,21,.90);
            box-shadow: 0 20px 80px rgba(0,0,0,.28);
            margin-bottom: 1.15rem;
        }}

        .eyebrow {{
            display: inline-flex;
            gap: .45rem;
            align-items: center;
            color: {LAVENDER};
            border: 1px solid rgba(196,171,236,.25);
            background: rgba(196,171,236,.07);
            border-radius: 999px;
            padding: .32rem .68rem;
            font-size: .76rem;
            font-weight: 750;
            letter-spacing: .09em;
            text-transform: uppercase;
        }}

        .hero h1 {{
            margin: .8rem 0 .35rem 0;
            font-size: clamp(2rem, 5vw, 4.2rem);
            line-height: .98;
            max-width: 1000px;
        }}

        .hero p {{
            color: {MUTED};
            max-width: 1060px;
            margin: .72rem 0 0 0;
            font-size: clamp(.98rem, 1.4vw, 1.08rem);
        }}

        .palette {{
            display: flex;
            height: 7px;
            border-radius: 999px;
            overflow: hidden;
            margin-top: 1.35rem;
            width: min(430px, 100%);
            box-shadow: 0 0 0 1px rgba(255,255,255,.04);
        }}
        .palette span {{ flex: 1; }}

        .kpi {{
            min-height: 128px;
            border: 1px solid {BORDER};
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(27,23,33,.96), rgba(19,17,23,.96));
            padding: 1rem 1.05rem;
            box-shadow: 0 12px 34px rgba(0,0,0,.18);
        }}
        .kpi-label {{
            color: {MUTED};
            font-size: .78rem;
            font-weight: 700;
            letter-spacing: .04em;
            text-transform: uppercase;
        }}
        .kpi-value {{
            color: {TEXT};
            font-size: clamp(1.65rem, 3vw, 2.45rem);
            font-weight: 820;
            line-height: 1.05;
            margin-top: .45rem;
        }}
        .kpi-note {{
            color: {MUTED};
            font-size: .82rem;
            margin-top: .42rem;
        }}

        .section-note {{
            border-left: 3px solid {PURPLE};
            border-radius: 0 14px 14px 0;
            background: rgba(168,102,190,.07);
            color: {MUTED};
            padding: .82rem 1rem;
            margin: .65rem 0 1rem 0;
        }}

        .method-card {{
            border: 1px solid {BORDER};
            border-radius: 18px;
            padding: 1rem 1.05rem;
            background: rgba(22,19,27,.78);
            margin-bottom: .8rem;
        }}
        .method-card strong {{ color: {LAVENDER}; }}

        .score-card {{
            border: 1px solid rgba(168,102,190,.34);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            background: linear-gradient(145deg, rgba(129,31,130,.13), rgba(22,19,27,.84));
            margin: .35rem 0 1rem 0;
        }}

        .small-muted {{ color: {MUTED}; font-size: .85rem; }}
        .source-pill {{
            display:inline-block;
            margin: .15rem .25rem .15rem 0;
            padding:.25rem .55rem;
            border-radius:999px;
            border:1px solid {BORDER};
            background:rgba(196,171,236,.055);
            color:{MUTED};
            font-size:.76rem;
        }}

        div[data-baseweb="tab-list"] {{
            gap: .35rem;
            overflow-x: auto;
            scrollbar-width: thin;
            padding-bottom: .25rem;
        }}
        button[data-baseweb="tab"] {{
            flex-shrink: 0;
            border-radius: 999px;
            border: 1px solid rgba(196,171,236,.13);
            background: rgba(22,19,27,.68);
            padding-left: .85rem;
            padding-right: .85rem;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            border-color: rgba(168,102,190,.50);
            background: rgba(168,102,190,.14);
        }}

        div[data-testid="stDataFrame"] {{
            border: 1px solid {BORDER};
            border-radius: 16px;
            overflow: hidden;
        }}

        div[data-testid="stExpander"] {{
            border: 1px solid {BORDER};
            border-radius: 16px;
            background: rgba(22,19,27,.55);
        }}

        .stSelectbox label, .stMultiSelect label, .stTextInput label,
        .stNumberInput label, .stSlider label, .stRadio label {{
            color: {MUTED} !important;
            font-weight: 650 !important;
        }}

        .stDownloadButton button, .stButton button {{
            border-radius: 12px;
            border: 1px solid rgba(196,171,236,.24);
            background: rgba(168,102,190,.12);
            color: {TEXT};
        }}

        div[data-testid="stCaptionContainer"],
        div[data-testid="stCaptionContainer"] * {{
            color: #FFFFFF !important;
        }}

        /* ============================================================
        TEXTO BRANCO EM ELEMENTOS COM FUNDO ROXO
        ============================================================ */

        /* Itens selecionados dos multiselects */
        span[data-baseweb="tag"],
        span[data-baseweb="tag"] * {{
            color: #FFFFFF !important;
        }}

        /* Aba selecionada */
        button[data-baseweb="tab"][aria-selected="true"],
        button[data-baseweb="tab"][aria-selected="true"] * {{
            color: #FFFFFF !important;
        }}

        /* Botões com fundo roxo */
        .stDownloadButton button,
        .stDownloadButton button *,
        .stButton button,
        .stButton button * {{
            color: #FFFFFF !important;
        }}

        /* Cards roxos do score */
        .score-card,
        .score-card * {{
            color: #FFFFFF !important;
        }}

        /* Caixas informativas com fundo arroxeado */
        .section-note,
        .section-note * {{
            color: #FFFFFF !important;
        }}

        @media (max-width: 720px) {{
            .block-container {{
                padding-left: .7rem;
                padding-right: .7rem;
                padding-top: .65rem;
            }}
            .hero {{ border-radius: 18px; }}
            .kpi {{ min-height: 108px; }}
            .section-note {{ font-size: .91rem; }}
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS DE DADOS
# ============================================================
def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip().upper()
    s = re.sub(r"\bD['’`]\s*", "DO ", s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def app_caption(text):
    st.markdown(
        f'<div class="app-caption">{text}</div>',
        unsafe_allow_html=True,
    )


def br_int(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{int(round(value)):,}".replace(",", ".")


def br_num(value: float | int | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    text = f"{float(value):,.{decimals}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def parse_decimal_br(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan}),
        errors="coerce",
    )


def safe_rate(count: pd.Series, population: pd.Series, multiplier: int = 100_000) -> pd.Series:
    return np.where(population > 0, count / population * multiplier, np.nan)


@st.cache_data(show_spinner=False)
def load_data():
    base_raw = pd.read_csv(BASE_PATH, sep=";", encoding="utf-8-sig")
    fem_raw = pd.read_csv(FEM_SENASP_PATH, encoding="utf-8-sig")
    estupro_raw = pd.read_csv(ESTUPRO_PATH, encoding="utf-8-sig")
    lesfem_raw = pd.read_csv(LESFEM_PATH, encoding="utf-8-sig")

    base = base_raw.copy()

    # --------------------------------------------------------
    # DADOS IBGE
    # --------------------------------------------------------
    # Campos numéricos vindos do arquivo do IBGE. Ausências permanecem NaN:
    # não transformamos dado ausente em zero.
    ibge_numeric_cols = [
        "IBGE - Área Territorial - km² [2025]",
        "IBGE - População no último censo - pessoas [2022]",
        "IBGE - Densidade demográfica - hab/km² [2022]",
        "IBGE - População estimada - pessoas [2026]",
        "IBGE - Escolarização 6 a 14 anos - % [2022]",
        "IBGE - IDHM [2010]",
        "IBGE - Mortalidade infantil - óbitos por mil nascidos vivos [2025]",
        "IBGE - Total de receitas brutas realizadas - R$ [2025]",
        "IBGE - Total de despesas brutas empenhadas - R$ [2025]",
        "IBGE - PIB per capita - R$ [2023]",
    ]
    for col in ibge_numeric_cols:
        if col in base.columns:
            base[col] = parse_decimal_br(base[col])

    if "IBGE - Código" in base.columns:
        base["IBGE - Código"] = (
            base["IBGE - Código"]
            .astype("string")
            .str.replace(r"\.0$", "", regex=True)
        )

    # Denominador populacional principal usado pelo painel.
    base[POPULATION_COL] = pd.to_numeric(base[POPULATION_SOURCE_COL], errors="coerce")


    # Indicadores financeiros per capita combinam valores financeiros de 2025
    # com a população do Censo 2022. Essa diferença de referência temporal é
    # informada no painel e os indicadores são apenas exploratórios.
    receita_col = "IBGE - Total de receitas brutas realizadas - R$ [2025]"
    despesa_col = "IBGE - Total de despesas brutas empenhadas - R$ [2025]"
    base["IBGE - Receita por habitante - R$"] = np.where(
        base[POPULATION_COL] > 0,
        base[receita_col] / base[POPULATION_COL],
        np.nan,
    )
    base["IBGE - Despesa por habitante - R$"] = np.where(
        base[POPULATION_COL] > 0,
        base[despesa_col] / base[POPULATION_COL],
        np.nan,
    )
    base["IBGE - Saldo fiscal por habitante - R$"] = np.where(
        base[POPULATION_COL] > 0,
        (base[receita_col] - base[despesa_col]) / base[POPULATION_COL],
        np.nan,
    )

    # --------------------------------------------------------
    # VIOLÊNCIA — taxas municipais
    # --------------------------------------------------------
    base["TAXA LESÃO CORPORAL"] = parse_decimal_br(base["TAXA LESÃO CORPORAL"])
    base["TAXA AMEAÇA"] = parse_decimal_br(base["TAXA AMEAÇA"])

    # Lesão e ameaça já chegam à base consolidada como taxas por 100 mil.
    # Como a nova base não contém as contagens absolutas desses dois indicadores,
    # preservamos as taxas da fonte sem inventar contagens.
    base["Lesão corporal por 100 mil"] = base["TAXA LESÃO CORPORAL"]
    base["Ameaça por 100 mil"] = base["TAXA AMEAÇA"]

    count_cols = [
        "Quantidade de CAPS",
        "LESFEM - Feminicídios consumados 2025",
        "LESFEM - Tentativas de feminicídio 2025",
        "LESFEM - Total de registros 2025",
        "LESFEM - Registros fonte Imprensa 2025",
        "LESFEM - Registros fonte SINESP 2025",
        "SENASP - Feminicídios consumados 2025",
        "SENASP - Tentativas de feminicídio 2025",
        "SENASP - Total vítimas feminicídio/tentativa 2025",
    ]
    for col in count_cols:
        base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0)

    base["mun_norm"] = base["Município"].map(normalize_name)

    # SENASP: taxas calculadas com a população de referência do IBGE.
    base["SENASP - feminicídios por 100 mil"] = safe_rate(
        base["SENASP - Feminicídios consumados 2025"], base[POPULATION_COL]
    )
    base["SENASP - tentativas por 100 mil"] = safe_rate(
        base["SENASP - Tentativas de feminicídio 2025"], base[POPULATION_COL]
    )
    base["SENASP - total por 100 mil"] = safe_rate(
        base["SENASP - Total vítimas feminicídio/tentativa 2025"], base[POPULATION_COL]
    )

    # Aliases públicos usados no restante do painel. A identificação da fonte
    # fica concentrada na comparação com o LESFEM e na aba de metodologia/fontes.
    base["Feminicídios consumados 2025"] = base["SENASP - Feminicídios consumados 2025"]
    base["Tentativas de feminicídio 2025"] = base["SENASP - Tentativas de feminicídio 2025"]
    base["Feminicídio + tentativa 2025"] = base["SENASP - Total vítimas feminicídio/tentativa 2025"]
    base["Feminicídios por 100 mil"] = base["SENASP - feminicídios por 100 mil"]
    base["Tentativas de feminicídio por 100 mil"] = base["SENASP - tentativas por 100 mil"]
    base["Feminicídio + tentativa por 100 mil"] = base["SENASP - total por 100 mil"]

    # LESFEM não entra no score; taxa apenas para análises próprias da fonte.
    base["LESFEM - taxa por 100 mil"] = safe_rate(
        base["LESFEM - Total de registros 2025"], base[POPULATION_COL]
    )
    base["Diferença LESFEM − SENASP"] = (
        base["LESFEM - Total de registros 2025"]
        - base["SENASP - Total vítimas feminicídio/tentativa 2025"]
    )

    fem = fem_raw.copy()
    fem["data_referencia"] = pd.to_datetime(fem["data_referencia"], dayfirst=True, errors="coerce")
    fem["vitima"] = pd.to_numeric(fem["vitima"], errors="coerce").fillna(0)
    fem["mun_norm"] = fem["municipio"].map(normalize_name)
    fem["mes_num"] = fem["data_referencia"].dt.month

    estupro = estupro_raw.copy()
    estupro["data_referencia"] = pd.to_datetime(estupro["data_referencia"], dayfirst=True, errors="coerce")
    estupro["feminino"] = pd.to_numeric(estupro["feminino"], errors="coerce").fillna(0)
    estupro["mes_num"] = estupro["data_referencia"].dt.month

    # Base detalhada LESFEM: preserva os campos originais e adiciona
    # somente auxiliares necessários às análises.
    lesfem = lesfem_raw.copy()
    lesfem["Mês"] = pd.to_numeric(lesfem["Mês"], errors="coerce").astype("Int64")
    lesfem["Idade da vítima num"] = pd.to_numeric(lesfem["Idade da vítima"], errors="coerce")
    lesfem["Idade do agressor num"] = pd.to_numeric(lesfem["Idade do agressor"], errors="coerce")
    lesfem["mun_norm"] = lesfem["Município"].map(normalize_name)

    lesfem_aliases = {
        "DIAMANTE D OESTE": "DIAMANTE DO OESTE",
        "ITAPEJARA D OESTE": "ITAPEJARA DO OESTE",
        "LARANJEIRAS": "LARANJEIRAS DO SUL",
    }
    lesfem["mun_norm"] = lesfem["mun_norm"].replace(lesfem_aliases)

    municipality_lookup = base.set_index("mun_norm")["Município"].to_dict()
    lesfem["Município padronizado"] = lesfem["mun_norm"].map(municipality_lookup)
    lesfem["Município padronizado"] = lesfem["Município padronizado"].fillna(
        lesfem["Município"].astype(str).str.title()
    )

    territorial_cols = [
        "mun_norm",
        POPULATION_COL,
        "IBGE - População no último censo - pessoas [2022]",
        "Regional de Saúde",
        "Macrorregional de Saúde",
    ]
    lesfem = lesfem.merge(base[territorial_cols], on="mun_norm", how="left")

    return base, fem, estupro, lesfem, base_raw, fem_raw, estupro_raw, lesfem_raw


# ============================================================
# MAPA — CARREGAMENTO MAIS ROBUSTO
# ============================================================
def _extract_geo_code(props: dict) -> str:
    for key in ["codarea", "id", "CD_MUN", "CD_MUN7", "geocodigo", "code"]:
        value = props.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _prepare_geojson_names(geo: dict, code_to_name: dict[str, str] | None = None) -> dict:
    code_to_name = code_to_name or {}
    for feature in geo.get("features", []):
        props = feature.setdefault("properties", {})
        code = _extract_geo_code(props)
        name = (
            code_to_name.get(code)
            or props.get("name")
            or props.get("nome")
            or props.get("NM_MUN")
            or props.get("description")
            or ""
        )
        props["mun_norm"] = normalize_name(name)
        if code:
            props["ibge_code"] = code
    return geo


def _validate_municipal_geojson(
    geo: dict,
    expected_names: tuple[str, ...] = (),
) -> tuple[bool, str]:
    """Evita aceitar como mapa municipal uma malha que contenha só o contorno do Paraná."""
    if not isinstance(geo, dict) or geo.get("type") != "FeatureCollection":
        return False, "a resposta não é um FeatureCollection GeoJSON"

    features = geo.get("features") or []
    polygon_features = [
        f for f in features
        if (f.get("geometry") or {}).get("type") in {"Polygon", "MultiPolygon"}
    ]

    # O Paraná tem 399 municípios. Um retorno com 1 (ou poucas) feições é o
    # contorno do Estado, não uma malha municipal — isso era o que produzia
    # o grande bloco verde no gráfico.
    if len(polygon_features) < 300:
        return False, f"somente {len(polygon_features)} feições poligonais; esperava-se uma malha municipal"

    names = {
        f.get("properties", {}).get("mun_norm", "")
        for f in polygon_features
        if f.get("properties", {}).get("mun_norm", "")
    }
    if len(names) < 300:
        return False, f"somente {len(names)} municípios identificáveis por nome"

    if expected_names:
        expected = set(expected_names)
        matches = names & expected
        # Aceita pequenas diferenças de grafia, mas não uma malha incompatível.
        min_matches = min(350, max(250, int(len(expected) * 0.85)))
        if len(matches) < min_matches:
            return False, f"apenas {len(matches)} nomes coincidem com a base municipal"

    return True, f"{len(polygon_features)} feições municipais"


@st.cache_data(ttl=7 * 86_400, show_spinner=False)
def load_geojson(expected_names: tuple[str, ...] = ()):
    """
    Carrega uma MALHA MUNICIPAL do Paraná e valida o conteúdo antes de usá-lo.

    Ordem:
    1) arquivo local opcional;
    2) GeoJSON municipal público já estruturado em 399 feições;
    3) API de Malhas do IBGE como fallback.

    A validação é importante porque alguns retornos da API do IBGE podem
    representar somente o contorno estadual. Se esse retorno for entregue ao
    Plotly como se fossem municípios, aparece um grande bloco uniforme.
    """
    errors: list[str] = []

    def accept(geo: dict, source: str, code_to_name: dict[str, str] | None = None):
        geo = _prepare_geojson_names(geo, code_to_name)
        ok, diagnostic = _validate_municipal_geojson(geo, expected_names)
        if ok:
            return geo, source, diagnostic
        errors.append(f"{source}: {diagnostic}")
        return None

    # 1) Local: melhor opção para produção/offline.
    if LOCAL_GEOJSON_PATH.exists():
        try:
            with LOCAL_GEOJSON_PATH.open("r", encoding="utf-8") as f:
                geo = json.load(f)
            result = accept(geo, "Arquivo local")
            if result:
                return result
        except Exception as exc:
            errors.append(f"arquivo local: {exc}")

    # 2) GeoJSON municipal conhecido: cada município é uma Feature.
    for url in GEOJSON_FALLBACK_URLS:
        try:
            response = requests.get(
                url,
                timeout=35,
                headers={"User-Agent": "Mozilla/5.0 dashboard-violencia-pr/2025"},
            )
            response.raise_for_status()
            result = accept(response.json(), "GeoJSON municipal público")
            if result:
                return result
        except Exception as exc:
            errors.append(f"GeoJSON municipal: {exc}")

    # 3) IBGE como fallback adicional.
    code_to_name: dict[str, str] = {}
    try:
        r_names = requests.get(
            IBGE_LOCALIDADES_URL,
            timeout=25,
            headers={"User-Agent": "Mozilla/5.0 dashboard-violencia-pr/2025"},
        )
        r_names.raise_for_status()
        for item in r_names.json():
            code_to_name[str(item.get("id", ""))] = str(item.get("nome", ""))
    except Exception as exc:
        errors.append(f"IBGE localidades: {exc}")

    params_options = [
        {
            "intrarregiao": "municipio",
            "qualidade": "minima",
            "formato": "application/vnd.geo+json",
        },
        {
            "intrarregiao": "municipio",
            "qualidade": "minima",
            "formato": "application/json",
        },
    ]

    for url in IBGE_GEO_URLS:
        for params in params_options:
            try:
                response = requests.get(
                    url,
                    params=params,
                    timeout=40,
                    headers={
                        "Accept": "application/vnd.geo+json, application/json",
                        "User-Agent": "Mozilla/5.0 dashboard-violencia-pr/2025",
                    },
                )
                response.raise_for_status()
                geo = response.json()
                # TopoJSON não pode ser entregue diretamente ao px.choropleth.
                if geo.get("type") != "FeatureCollection":
                    errors.append(f"{url}: formato retornado {geo.get('type', 'desconhecido')} não é GeoJSON FeatureCollection")
                    continue
                result = accept(geo, "IBGE — API de Malhas", code_to_name)
                if result:
                    return result
            except Exception as exc:
                errors.append(f"{url}: {exc}")

    raise RuntimeError(" | ".join(errors[-7:]) or "Nenhuma fonte de geometria municipal respondeu.")


MESES = {
    1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez",
}


def kpi(label: str, value: str, note: str = ""):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, height: int = 430) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter, Arial, sans-serif"),
        margin=dict(l=12, r=12, t=54, b=24),
        hoverlabel=dict(bgcolor=CARD_2, font_color=TEXT, bordercolor=PURPLE),
        legend=dict(bgcolor="rgba(0,0,0,0)", title_font_color=MUTED),
        colorway=PALETTE,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, tickfont_color=MUTED, title_font_color=MUTED)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, tickfont_color=MUTED, title_font_color=MUTED)
    return fig


def csv_download_bytes(df: pd.DataFrame, sep: str = ";") -> bytes:
    return df.to_csv(index=False, sep=sep).encode("utf-8-sig")


PUBLIC_COLUMN_RENAMES = {
    "SENASP - Feminicídios consumados 2025": "Feminicídios consumados 2025",
    "SENASP - Tentativas de feminicídio 2025": "Tentativas de feminicídio 2025",
    "SENASP - Total vítimas feminicídio/tentativa 2025": "Feminicídio + tentativa 2025",
    "SENASP - feminicídios por 100 mil": "Feminicídios por 100 mil",
    "SENASP - tentativas por 100 mil": "Tentativas de feminicídio por 100 mil",
    "SENASP - total por 100 mil": "Feminicídio + tentativa por 100 mil",
}


def publicize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove o nome da fonte dos rótulos exibidos fora das abas de fontes/comparação."""
    out = df.copy()
    for source_col, public_col in PUBLIC_COLUMN_RENAMES.items():
        if source_col not in out.columns:
            continue
        if public_col in out.columns:
            out = out.drop(columns=[source_col])
        else:
            out = out.rename(columns={source_col: public_col})
    return out


def percentile_position(series: pd.Series) -> pd.Series:
    """
    Posição relativa de 0 a 100 no conjunto de municípios.

    - 100 = entre os maiores valores do indicador;
    - 0 = valor zero ou sem ocorrência no indicador;
    - empates recebem a posição média do grupo empatado.

    A transformação por percentil evita que indicadores com escalas numéricas
    diferentes dominem o score simplesmente por terem números maiores.
    """
    s = pd.to_numeric(series, errors="coerce")
    valid = s.notna()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if valid.sum() == 0:
        return out
    ranked = s[valid].rank(method="average", pct=True) * 100
    ranked = ranked.where(s[valid] > 0, 0.0)
    out.loc[valid] = ranked
    return out


# Indicadores elegíveis para o score. Todos são NÃO-LESFEM e entram no
# cálculo na mesma unidade: ocorrências por 100 mil habitantes.
SCORE_INDICATORS = {
    "Lesão corporal — por 100 mil habitantes": {
        "score_col": "Lesão corporal por 100 mil",
        "raw_col": "Lesão corporal por 100 mil",
        "short": "Lesão corporal",
        "unit": "por 100 mil",
        "source": "base consolidada / CAPE",
    },
    "Ameaça — por 100 mil habitantes": {
        "score_col": "Ameaça por 100 mil",
        "raw_col": "Ameaça por 100 mil",
        "short": "Ameaça",
        "unit": "por 100 mil",
        "source": "base consolidada / CAPE",
    },
    "Feminicídio consumado — por 100 mil habitantes": {
        "score_col": "Feminicídios por 100 mil",
        "raw_col": "Feminicídios consumados 2025",
        "short": "Feminicídio",
        "unit": "por 100 mil",
        "source": "base principal",
    },
    "Tentativa de feminicídio — por 100 mil habitantes": {
        "score_col": "Tentativas de feminicídio por 100 mil",
        "raw_col": "Tentativas de feminicídio 2025",
        "short": "Tentativa",
        "unit": "por 100 mil",
        "source": "base principal",
    },
    "Feminicídio + tentativa — por 100 mil habitantes": {
        "score_col": "Feminicídio + tentativa por 100 mil",
        "raw_col": "Feminicídio + tentativa 2025",
        "short": "Feminicídio + tentativa",
        "unit": "por 100 mil",
        "source": "base principal",
    },
}

DEFAULT_SCORE_INDICATORS = [
    "Lesão corporal — por 100 mil habitantes",
    "Ameaça — por 100 mil habitantes",
    "Feminicídio consumado — por 100 mil habitantes",
    "Tentativa de feminicídio — por 100 mil habitantes",
]

# Pesos analíticos padrão do score. Não são pesos oficiais das fontes.
DEFAULT_SCORE_WEIGHTS = {
    "Ameaça — por 100 mil habitantes": 1.0,
    "Lesão corporal — por 100 mil habitantes": 2.0,
    "Tentativa de feminicídio — por 100 mil habitantes": 4.0,
    "Feminicídio consumado — por 100 mil habitantes": 5.0,
    "Feminicídio + tentativa — por 100 mil habitantes": 4.5,
}


def compute_score(
    df: pd.DataFrame,
    selected: Iterable[str],
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    selected = list(selected)
    out = df.copy()
    if not selected:
        out["Score"] = np.nan
        return out

    weights = weights or {name: 1.0 for name in selected}
    component_cols = []
    weighted_components = []
    total_weight = 0.0

    for name in selected:
        meta = SCORE_INDICATORS[name]
        component = f"Score · {meta['short']}"
        out[component] = percentile_position(out[meta["score_col"]])
        component_cols.append(component)
        w = float(weights.get(name, 1.0))
        if w > 0:
            weighted_components.append(out[component] * w)
            total_weight += w

    if total_weight <= 0:
        out["Score"] = np.nan
    else:
        weighted_sum = sum(weighted_components)
        out["Score"] = weighted_sum / total_weight

    out["Posição no Paraná"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    valid_score = out["Score"].notna()
    if valid_score.any():
        out.loc[valid_score, "Posição no Paraná"] = (
            out.loc[valid_score, "Score"]
            .rank(method="min", ascending=False)
            .astype("Int64")
        )
    return out


def apply_territorial_filters(df: pd.DataFrame, key: str, show_population: bool = True) -> pd.DataFrame:
    with st.expander("Filtros territoriais e de contexto", expanded=False):
        c1, c2 = st.columns(2)
        region_opts = sorted(df["Regional de Saúde"].dropna().astype(str).unique())
        macro_opts = sorted(df["Macrorregional de Saúde"].dropna().astype(str).unique())
        with c1:
            regions = st.multiselect("Regional de Saúde", region_opts, key=f"{key}_reg")
        with c2:
            macros = st.multiselect("Macrorregional de Saúde", macro_opts, key=f"{key}_macro")

        c3, c4 = st.columns(2)
        with c3:
            delegacia = st.selectbox("Delegacia da Mulher", ["Todos", "Sim", "Não"], key=f"{key}_delegacia")
        with c4:
            caps = st.selectbox("CAPS", ["Todos", "Sim", "Não"], key=f"{key}_caps")

        min_pop = 0
        if show_population:
            min_pop = st.number_input(
                "População mínima de referência para exibir no ranking (o score continua calculado sobre os 399 municípios)",
                min_value=0,
                max_value=int(df[POPULATION_COL].max()),
                value=0,
                step=5_000,
                key=f"{key}_minpop",
            )

    out = df.copy()
    if regions:
        out = out[out["Regional de Saúde"].isin(regions)]
    if macros:
        out = out[out["Macrorregional de Saúde"].isin(macros)]
    if delegacia != "Todos":
        out = out[out["Possui Delegacia da Mulher?"].astype(str).str.casefold() == delegacia.casefold()]
    if caps != "Todos":
        out = out[out["Possui CAPS?"].astype(str).str.casefold() == caps.casefold()]
    if min_pop:
        out = out[out[POPULATION_COL] >= min_pop]
    return out


# ============================================================
# DADOS
# ============================================================
base, fem_senasp, estupro, lesfem, base_raw, fem_raw, estupro_raw, lesfem_raw = load_data()

lesfem_consumados = int((lesfem["Consumado ou Tentado"] == "Consumado").sum())
lesfem_tentativas = int((lesfem["Consumado ou Tentado"] == "Tentado").sum())
lesfem_total = int(len(lesfem))
senasp_consumados = int(base["SENASP - Feminicídios consumados 2025"].sum())
senasp_tentativas = int(base["SENASP - Tentativas de feminicídio 2025"].sum())
senasp_total = int(base["SENASP - Total vítimas feminicídio/tentativa 2025"].sum())
estupro_total = int(estupro["feminino"].sum())
diff_total = lesfem_total - senasp_total
diff_pct = (diff_total / senasp_total * 100) if senasp_total else np.nan
mun_divergentes = int((base["Diferença LESFEM − SENASP"] != 0).sum())

# ============================================================
# CABEÇALHO
# ============================================================
st.markdown(
    f"""
    <section class="hero">
        <div class="eyebrow">Paraná · 2025 · painel interativo</div>
        <h1>Violência contra a Mulher</h1>
        <p>
            Dashboard para <strong>exposição, comparação e exploração de dados públicos</strong> de 2025.
        
        
    </section>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# ABAS
# ============================================================
tabs = st.tabs([
    "Score e comparação",
    "Informações por município",
    "LESFEM",
    "Estupros",
    "Território e rede",
    "Contexto IBGE",
    "Bases completas",
    "Metodologia e fontes",
])

# ------------------------------------------------------------
# 1) SCORE + MAPA + COMPARAÇÃO + CRUZAMENTOS
# ------------------------------------------------------------
with tabs[0]:
    st.subheader("Score municipal interativo")
    app_caption(
        "Escolha quais indicadores entram na pontuação. O score compara cada município com os demais municípios do Paraná e varia de 0 a 100."
    )

    st.markdown(
        """
        <div class="score-card">
            <strong>Como o score funciona:</strong> todos os indicadores entram como taxas por
            <strong>100 mil habitantes</strong>. Lesão corporal e ameaça usam as taxas já disponibilizadas na base
            consolidada; feminicídio e tentativa são calculados com a <strong>população do Censo Demográfico IBGE 2022</strong>,
            último censo disponível. Depois, cada município recebe
            uma posição relativa de 0 a 100 em cada indicador e essas posições são combinadas por
            <strong>média ponderada</strong>. Pesos padrão: ameaça = 1, lesão corporal = 2, tentativa = 4 e feminicídio = 5.
        </div>
        """,
        unsafe_allow_html=True,
    )
    app_caption(
        "Referência populacional: Censo Demográfico IBGE 2022. Os registros de violência são de 2025; "
        "por isso, a diferença temporal do denominador deve ser considerada na interpretação das taxas."
    )

    selected_indicators = st.multiselect(
        "Indicadores que devem compor o score",
        options=list(SCORE_INDICATORS.keys()),
        default=DEFAULT_SCORE_INDICATORS,
        help="O LESFEM não aparece aqui de propósito. Ele é usado somente na aba de comparação entre fontes.",
    )

    if not selected_indicators:
        st.warning("Selecione pelo menos um indicador para calcular o score.")
        selected_indicators = DEFAULT_SCORE_INDICATORS.copy()

    if (
        "Feminicídio + tentativa — por 100 mil habitantes" in selected_indicators
        and "Feminicídio consumado — por 100 mil habitantes" in selected_indicators
        and "Tentativa de feminicídio — por 100 mil habitantes" in selected_indicators
    ):
        st.info(
            "Você selecionou o total e também seus dois componentes. Isso dá peso adicional a feminicídio/tentativa no score. "
            "Se quiser pesos equivalentes entre fenômenos, use os componentes ou apenas o total."
        )

    # O score é sempre ponderado. Os pesos abaixo são valores padrão do painel
    # e podem ser ajustados pelo usuário sem alterar as bases de origem.
    weights = {
        name: float(DEFAULT_SCORE_WEIGHTS.get(name, 1.0))
        for name in selected_indicators
    }

    with st.expander("Pesos do score", expanded=False):
        app_caption(
            "O score é sempre ponderado. Um peso 4 tem quatro vezes a influência de um peso 1 na combinação final das posições relativas. "
            "Os valores iniciais são uma escolha analítica do painel e podem ser alterados."
        )
        weight_cols = st.columns(2)
        for idx, name in enumerate(selected_indicators):
            with weight_cols[idx % 2]:
                weights[name] = st.slider(
                    SCORE_INDICATORS[name]["short"],
                    min_value=0.5,
                    max_value=6.0,
                    value=float(DEFAULT_SCORE_WEIGHTS.get(name, 1.0)),
                    step=0.5,
                    key=f"weight_{idx}_{normalize_name(name)}",
                    help=f"Peso padrão: {DEFAULT_SCORE_WEIGHTS.get(name, 1.0):g}",
                )

        weights_summary = " · ".join(
            f"{SCORE_INDICATORS[name]['short']}: {weights[name]:g}"
            for name in selected_indicators
        )
        app_caption(f"Pesos aplicados nesta análise: {weights_summary}")

    # O score estadual é calculado ANTES dos filtros territoriais.
    scored = compute_score(base, selected_indicators, weights)
    view = apply_territorial_filters(scored, "score", show_population=True)
    view = view.sort_values(["Score", POPULATION_COL], ascending=[False, False])

    if view.empty:
        st.warning("Nenhum município corresponde aos filtros selecionados.")
    else:
        top_row = view.iloc[0]
        municipalities_with_senasp = int((base["SENASP - Total vítimas feminicídio/tentativa 2025"] > 0).sum())
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi("Maior score no recorte", str(top_row["Município"]), f"Score {br_num(top_row['Score'], 1)} / 100")
        with c2:
            kpi("Municípios exibidos", br_int(len(view)), f"de {len(base)} municípios do Paraná")
        with c3:
            kpi("Feminicídio + tentativa", br_int(senasp_total), f"registros em {municipalities_with_senasp} municípios")
        with c4:
            kpi("Indicadores no score", br_int(len(selected_indicators)), "score ponderado · pesos configuráveis")

        st.write("")
        rank_min = 5 if len(view) >= 5 else 1
        rank_n = st.slider(
            "Quantos municípios mostrar no ranking",
            min_value=rank_min,
            max_value=min(50, len(view)),
            value=min(20, len(view)),
            step=5 if len(view) >= 10 else 1,
        )
        ranking = view.head(rank_n).copy()

        left, right = st.columns([1.05, 1])
        with left:
            rank_plot = ranking.sort_values("Score")
            fig_rank = px.bar(
                rank_plot,
                x="Score",
                y="Município",
                orientation="h",
                color="Score",
                color_continuous_scale=[GREEN, SAGE, LAVENDER, PURPLE, MAGENTA],
                range_color=(0, 100),
                hover_data={
                    "Posição no Paraná": True,
                    POPULATION_COL: ":,.0f",
                    "Feminicídio + tentativa por 100 mil": ":.2f",
                    "Lesão corporal por 100 mil": ":.2f",
                    "Ameaça por 100 mil": ":.2f",
                },
                title=f"{rank_n} maiores scores no recorte selecionado",
            )
            style_fig(fig_rank, max(430, 28 * rank_n + 120))
            fig_rank.update_layout(
                yaxis_title="",
                xaxis_title="Score relativo (0–100)",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_rank, use_container_width=True, config={"displayModeBar": False})

        with right:
            ranking_cols = [
                "Posição no Paraná",
                "Município",
                "Score",
                POPULATION_COL,
                "Lesão corporal por 100 mil",
                "Ameaça por 100 mil",
                "Feminicídios consumados 2025",
                "Tentativas de feminicídio 2025",
                "Feminicídio + tentativa por 100 mil",
            ]
            st.markdown("#### Ranking detalhado")
            st.dataframe(
                ranking[ranking_cols].style.format({
                    "Score": "{:.1f}",
                    POPULATION_COL: "{:,.0f}",
                    "Lesão corporal por 100 mil": "{:.2f}",
                    "Ameaça por 100 mil": "{:.2f}",
                    "Feminicídio + tentativa por 100 mil": "{:.2f}",
                }),
                use_container_width=True,
                hide_index=True,
                height=min(680, 80 + rank_n * 31),
            )
            st.download_button(
                "Baixar ranking com score",
                data=csv_download_bytes(publicize_columns(view.drop(columns=["mun_norm"], errors="ignore"))),
                file_name="ranking_score_violencia_mulher_pr_2025.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown(
            """
            <div class="section-note">
                <strong>Por que o ranking é proporcional?</strong> Todos os componentes entram no score em taxas por
                100 mil habitantes. Para feminicídio e tentativa, o denominador populacional é a população municipal do Censo Demográfico
                IBGE 2022, último censo disponível. Isso reduz o efeito do tamanho absoluto do município. Em municípios pequenos, porém, uma
                única ocorrência pode gerar uma taxa alta; por isso a população de referência deve sempre ser observada
                junto com o score.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------------------------
        # COMPARAÇÃO DIRETA
        # ---------------------------
        st.markdown("### Comparar municípios")
        all_municipalities = scored.sort_values("Score", ascending=False)["Município"].tolist()
        default_compare = view.head(min(3, len(view)))["Município"].tolist()
        compare_names = st.multiselect(
            "Escolha de 2 a 6 municípios",
            options=all_municipalities,
            default=default_compare,
            max_selections=6,
            key="compare_municipalities",
        )

        if len(compare_names) >= 2:
            comp_df = scored[scored["Município"].isin(compare_names)].copy()
            comp_df = comp_df.sort_values("Score", ascending=False)

            component_cols = [f"Score · {SCORE_INDICATORS[name]['short']}" for name in selected_indicators]
            radar_rows = []
            for _, row in comp_df.iterrows():
                for name in selected_indicators:
                    short = SCORE_INDICATORS[name]["short"]
                    radar_rows.append({
                        "Município": row["Município"],
                        "Indicador": short,
                        "Percentil": row[f"Score · {short}"],
                    })
            radar_df = pd.DataFrame(radar_rows)

            c_left, c_right = st.columns([1.05, 1])
            with c_left:
                fig_compare = px.bar(
                    radar_df,
                    x="Indicador",
                    y="Percentil",
                    color="Município",
                    barmode="group",
                    color_discrete_sequence=PALETTE,
                    title="Posição relativa de cada município nos indicadores escolhidos",
                    labels={"Percentil": "Posição relativa (0–100)"},
                )
                style_fig(fig_compare, 470)
                fig_compare.update_yaxes(range=[0, 100])
                st.plotly_chart(fig_compare, use_container_width=True, config={"displayModeBar": False})

            with c_right:
                display_cols = [
                    "Município",
                    "Score",
                    "Posição no Paraná",
                    POPULATION_COL,
                    "Lesão corporal por 100 mil",
                    "Ameaça por 100 mil",
                    "Feminicídios consumados 2025",
                    "Tentativas de feminicídio 2025",
                    "Feminicídios por 100 mil",
                    "Tentativas de feminicídio por 100 mil",
                    "Possui Delegacia da Mulher?",
                    "Possui CAPS?",
                    "Quantidade de CAPS",
                ]
                st.dataframe(
                    comp_df[display_cols].style.format({
                        "Score": "{:.1f}",
                        POPULATION_COL: "{:,.0f}",
                        "Lesão corporal por 100 mil": "{:.2f}",
                        "Ameaça por 100 mil": "{:.2f}",
                        "Feminicídios por 100 mil": "{:.2f}",
                        "Tentativas de feminicídio por 100 mil": "{:.2f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            app_caption("Selecione pelo menos dois municípios para comparar.")

        # ---------------------------
        # MAPA NA PRIMEIRA ABA
        # ---------------------------
        st.markdown("### Mapa municipal")
        app_caption("O mapa agora tenta a API oficial de Malhas do IBGE antes das fontes de fallback.")

        map_metric_options = {"Score composto": "Score"}
        for name in selected_indicators:
            meta = SCORE_INDICATORS[name]
            map_metric_options[f"{meta['short']} · posição relativa"] = f"Score · {meta['short']}"
        map_metric_options.update({
            "Lesão corporal · por 100 mil": "Lesão corporal por 100 mil",
            "Ameaça · por 100 mil": "Ameaça por 100 mil",
            "Feminicídio · por 100 mil": "Feminicídios por 100 mil",
            "Tentativa de feminicídio · por 100 mil": "Tentativas de feminicídio por 100 mil",
            "Feminicídio + tentativa · por 100 mil": "Feminicídio + tentativa por 100 mil",
            "Feminicídio + tentativa · casos": "Feminicídio + tentativa 2025",
            "IBGE · população Censo 2022": POPULATION_COL,
            "IBGE · densidade demográfica 2022": "IBGE - Densidade demográfica - hab/km² [2022]",
            "IBGE · IDHM 2010": "IBGE - IDHM [2010]",
            "IBGE · escolarização 6–14 anos 2022": "IBGE - Escolarização 6 a 14 anos - % [2022]",
            "IBGE · mortalidade infantil 2025": "IBGE - Mortalidade infantil - óbitos por mil nascidos vivos [2025]",
            "IBGE · PIB per capita 2023": "IBGE - PIB per capita - R$ [2023]",
        })

        map_label = st.selectbox("O que as cores do mapa devem mostrar?", list(map_metric_options.keys()))
        map_metric = map_metric_options[map_label]

        try:
            # Valida a malha contra TODOS os municípios da base, e não apenas
            # contra o recorte que estiver filtrado na tela.
            expected_geo_names = tuple(sorted(base["mun_norm"].dropna().astype(str).unique()))
            geo, geo_source, geo_diagnostic = load_geojson(expected_geo_names)

            geo_names = {
                f.get("properties", {}).get("mun_norm", "")
                for f in geo.get("features", [])
                if f.get("properties", {}).get("mun_norm", "")
            }

            map_df = view.copy()
            map_df[map_metric] = pd.to_numeric(map_df[map_metric], errors="coerce")
            matched_mask = map_df["mun_norm"].isin(geo_names)
            missing_geo = map_df.loc[~matched_mask, "Município"].tolist()
            map_plot_df = map_df.loc[matched_mask & map_df[map_metric].notna()].copy()

            if map_plot_df.empty:
                raise RuntimeError("Nenhum município do recorte pôde ser ligado à malha geográfica.")

            # Segurança adicional: se a base inteira estiver selecionada e houver
            # pouquíssimas correspondências, não desenha um mapa enganoso.
            if len(view) >= 300 and len(map_plot_df) < 250:
                raise RuntimeError(
                    f"Somente {len(map_plot_df)} de {len(view)} municípios foram associados à malha. "
                    "O mapa foi interrompido para evitar uma visualização incorreta."
                )

            fig_map = px.choropleth(
                map_plot_df,
                geojson=geo,
                locations="mun_norm",
                featureidkey="properties.mun_norm",
                color=map_metric,
                hover_name="Município",
                hover_data={
                    "mun_norm": False,
                    "Score": ":.1f",
                    "Posição no Paraná": True,
                    POPULATION_COL: ":,.0f",
                    "Lesão corporal por 100 mil": ":.2f",
                    "Ameaça por 100 mil": ":.2f",
                    "Feminicídio + tentativa por 100 mil": ":.2f",
                    "IBGE - Densidade demográfica - hab/km² [2022]": ":.2f",
                    "IBGE - IDHM [2010]": ":.3f",
                    "IBGE - PIB per capita - R$ [2023]": ":,.2f",
                    "Regional de Saúde": True,
                    "Possui Delegacia da Mulher?": True,
                    "Possui CAPS?": True,
                },
                color_continuous_scale=[GREEN, SAGE, LAVENDER, PURPLE, MAGENTA],
                title=map_label,
            )
            fig_map.update_traces(
                marker_line_color=BG_2,
                marker_line_width=0.45,
            )
            fig_map.update_geos(
                fitbounds="locations",
                visible=False,
                bgcolor="rgba(0,0,0,0)",
                projection_type="mercator",
            )
            style_fig(fig_map, 690)
            fig_map.update_layout(
                coloraxis_colorbar=dict(
                    title=dict(text="Valor", font=dict(color=MUTED)),
                    tickfont=dict(color=MUTED),
                    thickness=12,
                )
            )
            st.plotly_chart(
                fig_map,
                use_container_width=True,
                config={"displayModeBar": False, "scrollZoom": True},
            )
            app_caption(
                f"Geometria: {geo_source} ({geo_diagnostic}). "
                f"Municípios desenhados neste recorte: {len(map_plot_df)}."
            )
            if missing_geo:
                st.warning(
                    f"{len(missing_geo)} município(s) do recorte não tiveram correspondência geográfica: "
                    + ", ".join(missing_geo[:15])
                )
        except Exception as exc:
            st.error(
                "Não foi possível montar corretamente a malha municipal do Paraná. "
                "O mapa foi interrompido em vez de exibir um bloco ou uma geometria incorreta."
            )
            with st.expander("Detalhes técnicos do mapa"):
                st.code(str(exc))

        # ---------------------------
        # CRUZAMENTO LIVRE
        # ---------------------------
        st.markdown("### Cruzar dados")
        app_caption("Escolha os eixos para investigar relações entre indicadores e contexto municipal. O gráfico é exploratório e não implica causalidade.")

        cross_options = {
            "Score composto": "Score",
            POPULATION_COL: POPULATION_COL,
            "Lesão corporal · por 100 mil": "Lesão corporal por 100 mil",
            "Ameaça · por 100 mil": "Ameaça por 100 mil",
            "Feminicídio · casos": "Feminicídios consumados 2025",
            "Tentativa de feminicídio · casos": "Tentativas de feminicídio 2025",
            "Feminicídio · por 100 mil": "Feminicídios por 100 mil",
            "Tentativa de feminicídio · por 100 mil": "Tentativas de feminicídio por 100 mil",
            "Feminicídio + tentativa · por 100 mil": "Feminicídio + tentativa por 100 mil",
            "Quantidade de CAPS": "Quantidade de CAPS",
            "IBGE · população Censo 2022": POPULATION_COL,
            "IBGE · área territorial (km²)": "IBGE - Área Territorial - km² [2025]",
            "IBGE · densidade demográfica": "IBGE - Densidade demográfica - hab/km² [2022]",
            "IBGE · escolarização 6–14 anos (%)": "IBGE - Escolarização 6 a 14 anos - % [2022]",
            "IBGE · IDHM": "IBGE - IDHM [2010]",
            "IBGE · mortalidade infantil": "IBGE - Mortalidade infantil - óbitos por mil nascidos vivos [2025]",
            "IBGE · PIB per capita": "IBGE - PIB per capita - R$ [2023]",
            "IBGE · receita por habitante": "IBGE - Receita por habitante - R$",
            "IBGE · despesa por habitante": "IBGE - Despesa por habitante - R$",
        }
        cx1, cx2, cx3 = st.columns(3)
        with cx1:
            x_label = st.selectbox("Eixo X", list(cross_options.keys()), index=0)
        with cx2:
            y_label = st.selectbox("Eixo Y", list(cross_options.keys()), index=2)
        with cx3:
            color_by = st.selectbox(
                "Cor por",
                ["Macrorregional de Saúde", "Possui Delegacia da Mulher?", "Possui CAPS?"],
            )

        x_col = cross_options[x_label]
        y_col = cross_options[y_label]

        # O tamanho das bolhas usa população. Alguns municípios da base não possuem
        # população preenchida; Plotly não aceita NaN em marker.size. Em vez de
        # inventar um tamanho, esses registros são omitidos apenas deste gráfico.
        cross_df = view.copy()
        for col in {x_col, y_col, POPULATION_COL}:
            cross_df[col] = pd.to_numeric(cross_df[col], errors="coerce")
        cross_df = cross_df.replace([np.inf, -np.inf], np.nan)

        before_cross = len(cross_df)
        cross_df = cross_df.dropna(subset=[x_col, y_col, POPULATION_COL]).copy()
        cross_df = cross_df.loc[cross_df[POPULATION_COL] >= 0].copy()
        omitted_cross = before_cross - len(cross_df)

        if cross_df.empty:
            st.warning("Não há municípios com dados válidos suficientes para formar este cruzamento.")
        else:
            fig_cross = px.scatter(
                cross_df,
                x=x_col,
                y=y_col,
                color=color_by,
                size=POPULATION_COL,
                size_max=34,
                color_discrete_sequence=PALETTE,
                hover_name="Município",
                hover_data={
                    "Score": ":.1f",
                    POPULATION_COL: ":,.0f",
                    "Regional de Saúde": True,
                    "Feminicídios consumados 2025": True,
                    "Tentativas de feminicídio 2025": True,
                },
                title=f"{x_label} × {y_label}",
            )
            style_fig(fig_cross, 540)
            st.plotly_chart(fig_cross, use_container_width=True, config={"displayModeBar": False})
            if omitted_cross:
                app_caption(
                    f"{omitted_cross} município(s) não aparecem neste cruzamento porque faltam dados "
                    "necessários ao eixo selecionado ou à população IBGE usada no tamanho das bolhas."
                )

# ------------------------------------------------------------
# 2) INFORMAÇÕES POR MUNICÍPIO
# ------------------------------------------------------------
with tabs[1]:
    st.subheader("Informações por município")
    app_caption(
        "Selecione um município para reunir em uma única tela os indicadores de violência, score, posição estadual, "
        "rede de atendimento, território e contexto socioeconômico."
    )

    municipality_options = sorted(scored["Município"].dropna().astype(str).unique().tolist())
    default_municipality = (
        scored.sort_values(["Score", POPULATION_COL], ascending=[False, False]).iloc[0]["Município"]
        if not scored.empty
        else municipality_options[0]
    )
    default_index = municipality_options.index(default_municipality) if default_municipality in municipality_options else 0

    selected_municipality = st.selectbox(
        "Selecione o município",
        options=municipality_options,
        index=default_index,
        key="municipality_profile_selector",
    )

    municipality_row = scored.loc[scored["Município"] == selected_municipality].iloc[0]
    municipality_norm = municipality_row["mun_norm"]

    st.markdown(
        f"""
        <div class="score-card">
            <strong>{selected_municipality}</strong><br>
            {municipality_row['Regional de Saúde']} · Macrorregional {municipality_row['Macrorregional de Saúde']}<br>
            <span style="color:{MUTED}">O score exibido abaixo usa exatamente os indicadores e pesos escolhidos na primeira aba.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        kpi(
            "Score atual",
            br_num(municipality_row["Score"], 1),
            "de 0 a 100 · configuração atual",
        )
    with p2:
        rank_value = municipality_row["Posição no Paraná"]
        rank_text = f"{int(rank_value)}º" if pd.notna(rank_value) else "—"
        kpi("Posição no Paraná", rank_text, f"entre {len(scored)} municípios")
    with p3:
        kpi("População de referência", br_int(municipality_row[POPULATION_COL]), "IBGE · Censo 2022")
    with p4:
        kpi(
            "Feminicídio + tentativa",
            br_int(municipality_row["Feminicídio + tentativa 2025"]),
            f"{br_num(municipality_row['Feminicídio + tentativa por 100 mil'], 2)} por 100 mil",
        )

    st.markdown("### Indicadores de violência")
    v1, v2, v3, v4 = st.columns(4)
    with v1:
        kpi(
            "Lesão corporal",
            br_num(municipality_row["Lesão corporal por 100 mil"], 2),
            "por 100 mil habitantes · base consolidada",
        )
    with v2:
        kpi(
            "Ameaça",
            br_num(municipality_row["Ameaça por 100 mil"], 2),
            "por 100 mil habitantes · base consolidada",
        )
    with v3:
        kpi(
            "Feminicídios consumados",
            br_int(municipality_row["Feminicídios consumados 2025"]),
            f"{br_num(municipality_row['Feminicídios por 100 mil'], 2)} por 100 mil",
        )
    with v4:
        kpi(
            "Tentativas de feminicídio",
            br_int(municipality_row["Tentativas de feminicídio 2025"]),
            f"{br_num(municipality_row['Tentativas de feminicídio por 100 mil'], 2)} por 100 mil",
        )

    st.markdown("#### Posição relativa nos indicadores escolhidos para o score")
    score_profile_rows = []
    for indicator_name in selected_indicators:
        meta = SCORE_INDICATORS[indicator_name]
        score_profile_rows.append({
            "Indicador": meta["short"],
            "Posição relativa": municipality_row[f"Score · {meta['short']}"],
            "Peso": float(weights.get(indicator_name, 1.0)),
        })
    score_profile_df = pd.DataFrame(score_profile_rows)

    if not score_profile_df.empty:
        fig_profile_score = px.bar(
            score_profile_df,
            x="Posição relativa",
            y="Indicador",
            orientation="h",
            color="Posição relativa",
            color_continuous_scale=[GREEN, SAGE, LAVENDER, PURPLE, MAGENTA],
            range_color=(0, 100),
            title=f"{selected_municipality} · posição relativa no Paraná",
            hover_data={"Peso": ":.2f"},
        )
        style_fig(fig_profile_score, max(330, 95 + 58 * len(score_profile_df)))
        fig_profile_score.update_layout(
            xaxis_title="Posição relativa (0–100)",
            yaxis_title="",
            coloraxis_showscale=False,
        )
        fig_profile_score.update_xaxes(range=[0, 100])
        st.plotly_chart(fig_profile_score, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        """
        <div class="section-note">
            A posição relativa mostra onde o município se encontra diante dos demais municípios do Paraná em cada
            indicador selecionado. Valor próximo de 100 significa estar entre os maiores valores daquele indicador.
            Essa posição compõe o score, mas não representa uma classificação oficial de segurança pública.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Evolução mensal · feminicídios e tentativas")
    municipality_monthly = fem_senasp.loc[fem_senasp["mun_norm"] == municipality_norm].copy()
    municipality_monthly = (
        municipality_monthly.groupby(["mes_num", "evento"], as_index=False)["vitima"]
        .sum()
        .sort_values("mes_num")
    )
    if not municipality_monthly.empty:
        municipality_monthly["Mês"] = municipality_monthly["mes_num"].map(MESES)
        fig_municipality_month = px.line(
            municipality_monthly,
            x="Mês",
            y="vitima",
            color="evento",
            markers=True,
            color_discrete_map={"Feminicídio": MAGENTA, "Tentativa de feminicídio": LAVENDER},
            title=f"Feminicídios e tentativas em {selected_municipality} · 2025",
            labels={"vitima": "Vítimas", "evento": "Evento"},
        )
        fig_municipality_month.update_traces(line=dict(width=3), marker=dict(size=8))
        style_fig(fig_municipality_month, 420)
        st.plotly_chart(fig_municipality_month, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Não há registros mensais disponíveis para este município no arquivo carregado.")

    st.markdown("### Território e rede de atendimento")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        kpi("Regional de Saúde", str(municipality_row["Regional de Saúde"]), "recorte territorial")
    with t2:
        kpi("Macrorregional", str(municipality_row["Macrorregional de Saúde"]), "recorte territorial")
    with t3:
        kpi("Delegacia da Mulher", str(municipality_row["Possui Delegacia da Mulher?"]), "presença no município")
    with t4:
        kpi(
            "CAPS",
            br_int(municipality_row["Quantidade de CAPS"]),
            f"Possui CAPS: {municipality_row['Possui CAPS?']}",
        )

    caps_modalities = municipality_row.get("Modalidades de CAPS", "")
    if pd.notna(caps_modalities) and str(caps_modalities).strip():
        st.markdown(f"**Modalidades de CAPS:** {caps_modalities}")
    else:
        app_caption("Não há modalidades de CAPS informadas para este município na base consolidada.")

    st.markdown("### Perfil demográfico e socioeconômico · IBGE")
    ib1, ib2, ib3, ib4 = st.columns(4)
    with ib1:
        kpi(
            "População Censo 2022",
            br_int(municipality_row[POPULATION_COL]),
            "IBGE · último censo disponível",
        )
    with ib2:
        kpi(
            "Densidade demográfica",
            br_num(municipality_row["IBGE - Densidade demográfica - hab/km² [2022]"], 2),
            "hab./km² · Censo 2022",
        )
    with ib3:
        kpi(
            "PIB per capita",
            f"R$ {br_num(municipality_row['IBGE - PIB per capita - R$ [2023]'], 2)}",
            "IBGE · 2023",
        )
    with ib4:
        kpi(
            "IDHM",
            br_num(municipality_row["IBGE - IDHM [2010]"], 3),
            "IBGE · 2010",
        )

    ibge_profile = pd.DataFrame([
        ["Código IBGE", municipality_row.get("IBGE - Código", "—")],
        ["Gentílico", municipality_row.get("IBGE - Gentílico", "—")],
        ["Prefeito (2025)", municipality_row.get("IBGE - Prefeito [2025]", "—")],
        ["Área territorial (km²)", br_num(municipality_row.get("IBGE - Área Territorial - km² [2025]"), 3)],
        ["População Censo 2022", br_int(municipality_row.get(POPULATION_COL))],
        ["Densidade demográfica (hab./km²)", br_num(municipality_row.get("IBGE - Densidade demográfica - hab/km² [2022]"), 2)],
        ["Escolarização 6 a 14 anos (%)", br_num(municipality_row.get("IBGE - Escolarização 6 a 14 anos - % [2022]"), 2)],
        ["IDHM", br_num(municipality_row.get("IBGE - IDHM [2010]"), 3)],
        ["Mortalidade infantil (por mil nascidos vivos)", br_num(municipality_row.get("IBGE - Mortalidade infantil - óbitos por mil nascidos vivos [2025]"), 2)],
        ["PIB per capita (R$)", br_num(municipality_row.get("IBGE - PIB per capita - R$ [2023]"), 2)],
        ["Receitas brutas realizadas (R$)", br_num(municipality_row.get("IBGE - Total de receitas brutas realizadas - R$ [2025]"), 0)],
        ["Despesas brutas empenhadas (R$)", br_num(municipality_row.get("IBGE - Total de despesas brutas empenhadas - R$ [2025]"), 0)],
        ["Receita 2025 por habitante — base Censo 2022 (R$)", br_num(municipality_row.get("IBGE - Receita por habitante - R$"), 2)],
        ["Despesa 2025 por habitante — base Censo 2022 (R$)", br_num(municipality_row.get("IBGE - Despesa por habitante - R$"), 2)],
        ["Saldo fiscal 2025 por habitante — base Censo 2022 (R$)", br_num(municipality_row.get("IBGE - Saldo fiscal por habitante - R$"), 2)],
    ], columns=["Indicador IBGE", "Valor"])
    st.dataframe(ibge_profile, use_container_width=True, hide_index=True, height=430)

    st.markdown("### Comparação das fontes · LESFEM × base principal")
    app_caption(
        "Os dados LESFEM aparecem aqui apenas como comparação de fonte. Eles não entram no score nem nos demais índices gerais do painel."
    )

    source_comparison = pd.DataFrame({
        "Evento": ["Feminicídio consumado", "Tentativa de feminicídio", "Total"],
        "Base principal": [
            municipality_row["SENASP - Feminicídios consumados 2025"],
            municipality_row["SENASP - Tentativas de feminicídio 2025"],
            municipality_row["SENASP - Total vítimas feminicídio/tentativa 2025"],
        ],
        "LESFEM": [
            municipality_row["LESFEM - Feminicídios consumados 2025"],
            municipality_row["LESFEM - Tentativas de feminicídio 2025"],
            municipality_row["LESFEM - Total de registros 2025"],
        ],
    })
    source_comparison["Diferença LESFEM − base principal"] = source_comparison["LESFEM"] - source_comparison["Base principal"]

    sc_left, sc_right = st.columns([1.1, 1])
    with sc_left:
        source_long = source_comparison.melt(
            id_vars="Evento",
            value_vars=["Base principal", "LESFEM"],
            var_name="Fonte",
            value_name="Registros",
        )
        fig_source_mun = px.bar(
            source_long,
            x="Evento",
            y="Registros",
            color="Fonte",
            barmode="group",
            color_discrete_map={"Base principal": PURPLE, "LESFEM": SAGE},
            title=f"Comparação de registros em {selected_municipality}",
        )
        style_fig(fig_source_mun, 390)
        fig_source_mun.update_layout(xaxis_title="", yaxis_title="Registros")
        st.plotly_chart(fig_source_mun, use_container_width=True, config={"displayModeBar": False})
    with sc_right:
        st.dataframe(
            source_comparison.style.format({
                "Base principal": "{:,.0f}",
                "LESFEM": "{:,.0f}",
                "Diferença LESFEM − base principal": "{:+,.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(
            f"**Origem declarada dos registros LESFEM neste município:** "
            f"{br_int(municipality_row['LESFEM - Registros fonte Imprensa 2025'])} da imprensa e "
            f"{br_int(municipality_row['LESFEM - Registros fonte SINESP 2025'])} do SINESP."
        )

    st.markdown("### Ficha completa do município")
    app_caption("Todos os campos municipais disponíveis na base consolidada, organizados em uma única consulta.")

    complete_profile = pd.DataFrame([
        ["Município", municipality_row["Município"]],
        ["Código IBGE", municipality_row.get("IBGE - Código", "—")],
        ["Gentílico", municipality_row.get("IBGE - Gentílico", "—")],
        ["Prefeito (2025)", municipality_row.get("IBGE - Prefeito [2025]", "—")],
        ["Área territorial (km²)", br_num(municipality_row.get("IBGE - Área Territorial - km² [2025]"), 3)],
        ["População IBGE — Censo 2022", br_int(municipality_row.get(POPULATION_COL))],
        ["Densidade demográfica (hab./km²)", br_num(municipality_row.get("IBGE - Densidade demográfica - hab/km² [2022]"), 2)],
        ["Escolarização 6 a 14 anos (%)", br_num(municipality_row.get("IBGE - Escolarização 6 a 14 anos - % [2022]"), 2)],
        ["IDHM", br_num(municipality_row.get("IBGE - IDHM [2010]"), 3)],
        ["Mortalidade infantil", br_num(municipality_row.get("IBGE - Mortalidade infantil - óbitos por mil nascidos vivos [2025]"), 2)],
        ["PIB per capita (R$)", br_num(municipality_row.get("IBGE - PIB per capita - R$ [2023]"), 2)],
        ["Regional de Saúde", municipality_row["Regional de Saúde"]],
        ["Macrorregional de Saúde", municipality_row["Macrorregional de Saúde"]],
        ["Lesão corporal · por 100 mil", br_num(municipality_row["Lesão corporal por 100 mil"], 2)],
        ["Ameaça · por 100 mil", br_num(municipality_row["Ameaça por 100 mil"], 2)],
        ["Feminicídios consumados", br_int(municipality_row["Feminicídios consumados 2025"])],
        ["Tentativas de feminicídio", br_int(municipality_row["Tentativas de feminicídio 2025"])],
        ["Feminicídio + tentativa", br_int(municipality_row["Feminicídio + tentativa 2025"])],
        ["Feminicídios por 100 mil", br_num(municipality_row["Feminicídios por 100 mil"], 2)],
        ["Tentativas por 100 mil", br_num(municipality_row["Tentativas de feminicídio por 100 mil"], 2)],
        ["Feminicídio + tentativa por 100 mil", br_num(municipality_row["Feminicídio + tentativa por 100 mil"], 2)],
        ["Possui Delegacia da Mulher?", municipality_row["Possui Delegacia da Mulher?"]],
        ["Possui CAPS?", municipality_row["Possui CAPS?"]],
        ["Quantidade de CAPS", br_int(municipality_row["Quantidade de CAPS"])],
        ["Modalidades de CAPS", municipality_row["Modalidades de CAPS"]],
        ["Score atual", br_num(municipality_row["Score"], 1)],
        ["Posição no Paraná", rank_text],
        ["LESFEM · feminicídios consumados", br_int(municipality_row["LESFEM - Feminicídios consumados 2025"])],
        ["LESFEM · tentativas de feminicídio", br_int(municipality_row["LESFEM - Tentativas de feminicídio 2025"])],
        ["LESFEM · total de registros", br_int(municipality_row["LESFEM - Total de registros 2025"])],
        ["LESFEM · registros fonte Imprensa", br_int(municipality_row["LESFEM - Registros fonte Imprensa 2025"])],
        ["LESFEM · registros fonte SINESP", br_int(municipality_row["LESFEM - Registros fonte SINESP 2025"])],
        ["Diferença LESFEM − base principal", br_num(municipality_row["Diferença LESFEM − SENASP"], 0)],
    ], columns=["Campo", "Valor"])
    st.dataframe(complete_profile, use_container_width=True, hide_index=True, height=620)

    municipality_download = scored.loc[scored["Município"] == selected_municipality].drop(
        columns=["mun_norm"], errors="ignore"
    )
    st.download_button(
        "Baixar ficha deste município em CSV",
        data=csv_download_bytes(publicize_columns(municipality_download)),
        file_name=f"{normalize_name(selected_municipality).lower().replace(' ', '_')}_violencia_mulher_2025.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ------------------------------------------------------------
# 3) LESFEM — ANÁLISE DETALHADA + COMPARAÇÃO SECUNDÁRIA SENASP
# ------------------------------------------------------------
with tabs[2]:
    st.subheader("LESFEM · análise detalhada dos registros de 2025")
    

    st.markdown(
        """
        <div class="method-card">
            <strong>LESFEM - Laboratório de Estudos de Feminicídios:</strong><br>
            vinculado à Universidade Estadual de Londrina (UEL), desenvolve pesquisas e ações voltadas à produção e análise de dados sobre feminicídios e violência de gênero, incluindo o Monitor de Feminicídios no Brasil. Sua base pública de 2025 reúne casos de feminicídios consumados e tentados identificados a partir de duas fontes principais — imprensa e SINESP — e contém informações como município, desfecho do caso, vínculo entre vítima e suspeito, local do crime, meio utilizado e idade das pessoas envolvidas. A própria instituição destaca que a base possui finalidade analítica e de transparência, não substitui as bases oficiais e pode sofrer atualizações e reclassificações.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------
    # Filtros LESFEM
    # --------------------------
    with st.expander("Filtros da análise LESFEM", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            les_outcomes = st.multiselect(
                "Desfecho",
                options=["Consumado", "Tentado"],
                default=["Consumado", "Tentado"],
                key="les_outcomes",
            )
        with f2:
            les_sources = st.multiselect(
                "Origem declarada no LESFEM",
                options=sorted(lesfem["Fonte"].dropna().astype(str).unique().tolist()),
                default=sorted(lesfem["Fonte"].dropna().astype(str).unique().tolist()),
                key="les_sources",
            )
        with f3:
            les_month_range = st.slider(
                "Meses considerados",
                min_value=1,
                max_value=12,
                value=(1, 12),
                key="les_month_range",
            )

        les_municipalities = st.multiselect(
            "Municípios · deixe vazio para considerar todos",
            options=sorted(lesfem["Município padronizado"].dropna().astype(str).unique().tolist()),
            default=[],
            key="les_municipalities",
        )

    les_filtered = lesfem.copy()
    if les_outcomes:
        les_filtered = les_filtered[les_filtered["Consumado ou Tentado"].isin(les_outcomes)]
    else:
        les_filtered = les_filtered.iloc[0:0]
    if les_sources:
        les_filtered = les_filtered[les_filtered["Fonte"].isin(les_sources)]
    else:
        les_filtered = les_filtered.iloc[0:0]
    les_filtered = les_filtered[
        les_filtered["Mês"].between(les_month_range[0], les_month_range[1], inclusive="both")
    ]
    if les_municipalities:
        les_filtered = les_filtered[les_filtered["Município padronizado"].isin(les_municipalities)]

    if les_filtered.empty:
        st.warning("Nenhum registro LESFEM corresponde aos filtros escolhidos.")
    else:
        lf_total = int(len(les_filtered))
        lf_consumados = int((les_filtered["Consumado ou Tentado"] == "Consumado").sum())
        lf_tentados = int((les_filtered["Consumado ou Tentado"] == "Tentado").sum())
        lf_municipios = int(les_filtered["Município padronizado"].nunique())
        lf_imprensa = int((les_filtered["Fonte"] == "Imprensa").sum())
        lf_sinesp = int((les_filtered["Fonte"] == "SINESP").sum())

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi("Registros LESFEM", br_int(lf_total), "recorte atual dos filtros")
        with k2:
            kpi("Consumados", br_int(lf_consumados), f"{br_num(lf_consumados / lf_total * 100, 1)}% do recorte")
        with k3:
            kpi("Tentativas", br_int(lf_tentados), f"{br_num(lf_tentados / lf_total * 100, 1)}% do recorte")
        with k4:
            kpi("Municípios", br_int(lf_municipios), "com ao menos um registro")

        k5, k6 = st.columns(2)
        with k5:
            kpi("Origem · Imprensa", br_int(lf_imprensa), f"{br_num(lf_imprensa / lf_total * 100, 1)}%")
        with k6:
            kpi("Origem · SINESP", br_int(lf_sinesp), f"{br_num(lf_sinesp / lf_total * 100, 1)}%")

        # --------------------------
        # Evolução temporal + origem
        # --------------------------
        st.markdown("### Evolução dos registros ao longo de 2025")
        t1, t2 = st.columns([1.45, 1])
        with t1:
            les_monthly = (
                les_filtered.groupby(["Mês", "Consumado ou Tentado"], as_index=False)
                .size()
                .rename(columns={"size": "Registros"})
            )
            les_monthly["Mês nome"] = les_monthly["Mês"].map(MESES)
            les_monthly["ordem"] = les_monthly["Mês"].astype(int)
            les_monthly = les_monthly.sort_values("ordem")
            fig_les_month = px.bar(
                les_monthly,
                x="Mês nome",
                y="Registros",
                color="Consumado ou Tentado",
                barmode="group",
                category_orders={"Mês nome": list(MESES.values())},
                color_discrete_map={"Consumado": MAGENTA, "Tentado": LAVENDER},
                title="Registros mensais no LESFEM · consumados e tentativas",
                labels={"Mês nome": "Mês", "Consumado ou Tentado": "Desfecho"},
            )
            style_fig(fig_les_month, 430)
            st.plotly_chart(fig_les_month, use_container_width=True, config={"displayModeBar": False})

        with t2:
            source_outcome = (
                les_filtered.groupby(["Fonte", "Consumado ou Tentado"], as_index=False)
                .size()
                .rename(columns={"size": "Registros"})
            )
            fig_source_outcome = px.bar(
                source_outcome,
                x="Fonte",
                y="Registros",
                color="Consumado ou Tentado",
                barmode="stack",
                color_discrete_map={"Consumado": MAGENTA, "Tentado": LAVENDER},
                title="Desfecho segundo a origem declarada",
                labels={"Consumado ou Tentado": "Desfecho"},
            )
            style_fig(fig_source_outcome, 430)
            st.plotly_chart(fig_source_outcome, use_container_width=True, config={"displayModeBar": False})

        # --------------------------
        # Contexto dos casos
        # --------------------------
        st.markdown("### Contexto dos registros LESFEM")
        c1, c2 = st.columns(2)
        with c1:
            local_counts = (
                les_filtered["Local do crime"]
                .fillna("Incerto/Desconhecido")
                .astype(str)
                .value_counts()
                .rename_axis("Local do crime")
                .reset_index(name="Registros")
                .sort_values("Registros")
            )
            fig_local = px.bar(
                local_counts,
                x="Registros",
                y="Local do crime",
                orientation="h",
                title="Onde os crimes ocorreram",
                color="Registros",
                color_continuous_scale=[GREEN, SAGE, LAVENDER, PURPLE, MAGENTA],
            )
            style_fig(fig_local, 520)
            fig_local.update_layout(coloraxis_showscale=False, yaxis_title="")
            st.plotly_chart(fig_local, use_container_width=True, config={"displayModeBar": False})

        with c2:
            relation_counts = (
                les_filtered["Vínculo entre vítima e suspeito"]
                .fillna("Incerto/Desconhecido")
                .astype(str)
                .value_counts()
                .rename_axis("Vínculo")
                .reset_index(name="Registros")
                .sort_values("Registros")
            )
            fig_relation = px.bar(
                relation_counts,
                x="Registros",
                y="Vínculo",
                orientation="h",
                title="Vínculo entre vítima e suspeito",
                color="Registros",
                color_continuous_scale=[GREEN, SAGE, LAVENDER, PURPLE, MAGENTA],
            )
            style_fig(fig_relation, 520)
            fig_relation.update_layout(coloraxis_showscale=False, yaxis_title="")
            st.plotly_chart(fig_relation, use_container_width=True, config={"displayModeBar": False})

        c3, c4 = st.columns(2)
        with c3:
            weapon_counts = (
                les_filtered["Arma ou meio utilizado"]
                .fillna("Incerto/Desconhecido")
                .astype(str)
                .value_counts()
                .rename_axis("Arma ou meio")
                .reset_index(name="Registros")
                .sort_values("Registros")
            )
            fig_weapon = px.bar(
                weapon_counts,
                x="Registros",
                y="Arma ou meio",
                orientation="h",
                title="Arma ou meio utilizado",
                color="Registros",
                color_continuous_scale=[GREEN, SAGE, LAVENDER, PURPLE, MAGENTA],
            )
            style_fig(fig_weapon, 520)
            fig_weapon.update_layout(coloraxis_showscale=False, yaxis_title="")
            st.plotly_chart(fig_weapon, use_container_width=True, config={"displayModeBar": False})

        with c4:
            period_counts = (
                les_filtered["Período do dia"]
                .fillna("Incerto/Desconhecido")
                .astype(str)
                .value_counts()
                .rename_axis("Período")
                .reset_index(name="Registros")
            )
            fig_period = px.pie(
                period_counts,
                names="Período",
                values="Registros",
                hole=.56,
                color="Período",
                color_discrete_sequence=PALETTE,
                title="Período do dia",
            )
            fig_period.update_traces(textinfo="percent+value")
            style_fig(fig_period, 520)
            st.plotly_chart(fig_period, use_container_width=True, config={"displayModeBar": False})

        # --------------------------
        # Perfil etário + dependentes + completude
        # --------------------------
        st.markdown("### Perfil e completude das informações")
        p1, p2 = st.columns([1.25, 1])
        with p1:
            age_v = les_filtered.loc[les_filtered["Idade da vítima num"].notna(), ["Idade da vítima num"]].copy()
            age_v.columns = ["Idade"]
            age_v["Perfil"] = "Vítima"
            age_a = les_filtered.loc[les_filtered["Idade do agressor num"].notna(), ["Idade do agressor num"]].copy()
            age_a.columns = ["Idade"]
            age_a["Perfil"] = "Agressor"
            age_long = pd.concat([age_v, age_a], ignore_index=True)

            if not age_long.empty:
                fig_age = px.histogram(
                    age_long,
                    x="Idade",
                    color="Perfil",
                    barmode="overlay",
                    nbins=18,
                    opacity=.72,
                    color_discrete_map={"Vítima": PURPLE, "Agressor": SAGE},
                    title="Distribuição das idades informadas",
                )
                style_fig(fig_age, 410)
                st.plotly_chart(fig_age, use_container_width=True, config={"displayModeBar": False})
            app_caption(
                f"Idade informada para {br_int(len(age_v))} vítimas e {br_int(len(age_a))} agressores no recorte atual."
            )

        with p2:
            dep_counts = (
                les_filtered["Filhos dependentes (informação disponível?)"]
                .fillna("Incerto/Desconhecido")
                .astype(str)
                .value_counts()
                .rename_axis("Informação")
                .reset_index(name="Registros")
            )
            fig_dep = px.bar(
                dep_counts,
                x="Informação",
                y="Registros",
                color="Informação",
                color_discrete_sequence=PALETTE,
                title="Informação disponível sobre filhos dependentes",
            )
            style_fig(fig_dep, 410)
            fig_dep.update_layout(showlegend=False, xaxis_title="")
            st.plotly_chart(fig_dep, use_container_width=True, config={"displayModeBar": False})

        def known_text(series: pd.Series) -> pd.Series:
            text = series.fillna("").astype(str).str.strip()
            return ~text.str.casefold().isin({"", "na", "nan", "incerto/desconhecido"})

        completeness = pd.DataFrame({
            "Campo": [
                "Local do crime",
                "Vínculo vítima–suspeito",
                "Arma ou meio",
                "Filhos dependentes",
                "Período do dia",
                "Idade da vítima",
                "Idade do agressor",
            ],
            "Percentual informado": [
                known_text(les_filtered["Local do crime"]).mean() * 100,
                known_text(les_filtered["Vínculo entre vítima e suspeito"]).mean() * 100,
                known_text(les_filtered["Arma ou meio utilizado"]).mean() * 100,
                known_text(les_filtered["Filhos dependentes (informação disponível?)"]).mean() * 100,
                known_text(les_filtered["Período do dia"]).mean() * 100,
                les_filtered["Idade da vítima num"].notna().mean() * 100,
                les_filtered["Idade do agressor num"].notna().mean() * 100,
            ],
        }).sort_values("Percentual informado")
        fig_complete = px.bar(
            completeness,
            x="Percentual informado",
            y="Campo",
            orientation="h",
            range_x=[0, 100],
            text="Percentual informado",
            color="Percentual informado",
            color_continuous_scale=[GREEN, SAGE, LAVENDER, PURPLE],
            title="Completude dos principais campos da base",
            labels={"Percentual informado": "% com informação específica"},
        )
        fig_complete.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        style_fig(fig_complete, 430)
        fig_complete.update_layout(coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig_complete, use_container_width=True, config={"displayModeBar": False})
        app_caption(
            "Para este gráfico, NA, vazio e Incerto/Desconhecido são tratados como informação não especificada. Eles continuam preservados na base e nos demais gráficos."
        )

        # --------------------------
        # Municípios na base LESFEM
        # --------------------------
        st.markdown("### Municípios na base LESFEM")
        mun_les = (
            les_filtered.groupby(
                ["Município padronizado", POPULATION_COL],
                dropna=False,
                as_index=False,
            )
            .agg(
                Registros=("Identificador", "count"),
                Consumados=("Consumado ou Tentado", lambda s: int((s == "Consumado").sum())),
                Tentativas=("Consumado ou Tentado", lambda s: int((s == "Tentado").sum())),
            )
        )
        mun_les["Taxa LESFEM por 100 mil"] = np.where(
            mun_les[POPULATION_COL] > 0,
            mun_les["Registros"] / mun_les[POPULATION_COL] * 100_000,
            np.nan,
        )

        mr1, mr2 = st.columns([1, 1])
        with mr1:
            les_rank_metric = st.radio(
                "Ordenar municípios por",
                ["Número de registros", "Taxa por 100 mil habitantes"],
                horizontal=True,
                key="les_rank_metric",
            )
        with mr2:
            les_min_pop = st.number_input(
                "População mínima para o ranking proporcional",
                min_value=0,
                max_value=int(base[POPULATION_COL].max()),
                value=0,
                step=5_000,
                key="les_min_pop",
                help="Útil para reduzir oscilações extremas de taxas em municípios muito pequenos.",
            )

        mun_rank = mun_les.copy()
        if les_min_pop:
            mun_rank = mun_rank[mun_rank[POPULATION_COL] >= les_min_pop]
        metric_col = "Registros" if les_rank_metric == "Número de registros" else "Taxa LESFEM por 100 mil"
        mun_rank = mun_rank.dropna(subset=[metric_col]).nlargest(18, metric_col).sort_values(metric_col)

        fig_mun_les = px.bar(
            mun_rank,
            x=metric_col,
            y="Município padronizado",
            orientation="h",
            color=metric_col,
            color_continuous_scale=[GREEN, SAGE, LAVENDER, PURPLE, MAGENTA],
            title=f"Municípios com maiores valores · {les_rank_metric.lower()}",
            hover_data=["Registros", "Consumados", "Tentativas", POPULATION_COL],
        )
        style_fig(fig_mun_les, 620)
        fig_mun_les.update_layout(coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig_mun_les, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            """
            <div class="section-note">
                A taxa por 100 mil habitantes nesta seção é uma medida descritiva calculada a partir dos registros
                LESFEM e da população do Censo IBGE 2022 usada como referência no painel. Em municípios pequenos, poucos registros podem
                produzir taxas elevadas; por isso o painel permite definir uma população mínima para o ranking.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(
            mun_les.sort_values(["Registros", "Município padronizado"], ascending=[False, True]).style.format({
                POPULATION_COL: "{:,.0f}",
                "Taxa LESFEM por 100 mil": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
            height=420,
        )

        # --------------------------
        # Detalhe municipal LESFEM
        # --------------------------
        st.markdown("### Detalhar um município na base LESFEM")
        les_detail_mun = st.selectbox(
            "Município",
            options=sorted(lesfem["Município padronizado"].dropna().astype(str).unique().tolist()),
            key="les_detail_mun",
        )
        les_one = lesfem[lesfem["Município padronizado"] == les_detail_mun].copy()
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            kpi("Registros", br_int(len(les_one)), les_detail_mun)
        with d2:
            kpi("Consumados", br_int((les_one["Consumado ou Tentado"] == "Consumado").sum()), "LESFEM")
        with d3:
            kpi("Tentativas", br_int((les_one["Consumado ou Tentado"] == "Tentado").sum()), "LESFEM")
        with d4:
            one_pop = les_one[POPULATION_COL].dropna()
            one_rate = (len(les_one) / one_pop.iloc[0] * 100_000) if not one_pop.empty and one_pop.iloc[0] > 0 else np.nan
            kpi("Taxa LESFEM", br_num(one_rate, 2), "por 100 mil habitantes")

        detail_cols = [
            "Identificador", "Fonte", "Mês", "Consumado ou Tentado", "Local do crime",
            "Idade da vítima", "Vínculo entre vítima e suspeito", "Arma ou meio utilizado",
            "Filhos dependentes (informação disponível?)", "Idade do agressor", "Período do dia",
        ]
        st.dataframe(les_one[detail_cols], use_container_width=True, hide_index=True, height=360)

    # ========================================================
    # COMPARAÇÃO SECUNDÁRIA COM SENASP
    # ========================================================
    st.markdown("---")
    st.markdown("### Comparação secundária · LESFEM × SENASP/Sinesp")
    st.markdown(
        """
        <div class="section-note">
            Esta comparação não funde as bases nem define qual delas é a “correta”. Ela mostra a diferença entre
            registros produzidos por metodologias e fluxos distintos. O LESFEM permanece como objeto principal desta aba.
        </div>
        """,
        unsafe_allow_html=True,
    )

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        kpi("LESFEM · total", br_int(lesfem_total), "consumados + tentativas")
    with sc2:
        kpi("SENASP · total", br_int(senasp_total), "consumados + tentativas")
    with sc3:
        diff_sign = "+" if diff_total >= 0 else ""
        kpi("Diferença", f"{diff_sign}{br_int(diff_total)}", "LESFEM − SENASP")
    with sc4:
        kpi("Municípios divergentes", br_int(mun_divergentes), f"de {len(base)} municípios")

    cc1, cc2 = st.columns(2)
    with cc1:
        comp_detail = pd.DataFrame({
            "Indicador": ["Consumados", "Tentativas", "Total"],
            "LESFEM": [lesfem_consumados, lesfem_tentativas, lesfem_total],
            "SENASP": [senasp_consumados, senasp_tentativas, senasp_total],
        })
        comp_long = comp_detail.melt(
            id_vars="Indicador", value_vars=["LESFEM", "SENASP"],
            var_name="Base", value_name="Registros"
        )
        fig_comp = px.bar(
            comp_long,
            x="Indicador",
            y="Registros",
            color="Base",
            barmode="group",
            color_discrete_map={"LESFEM": PURPLE, "SENASP": SAGE},
            title="Totais estaduais nas duas bases",
        )
        style_fig(fig_comp, 420)
        st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})

    with cc2:
        les_month_total = lesfem.groupby("Mês", as_index=False).size().rename(columns={"size": "Registros"})
        les_month_total["Base"] = "LESFEM"
        sen_month_total = (
            fem_senasp.groupby("mes_num", as_index=False)["vitima"]
            .sum()
            .rename(columns={"mes_num": "Mês", "vitima": "Registros"})
        )
        sen_month_total["Base"] = "SENASP"
        month_compare = pd.concat([les_month_total, sen_month_total], ignore_index=True)
        month_compare["Mês nome"] = month_compare["Mês"].map(MESES)
        month_compare = month_compare.sort_values(["Mês", "Base"])
        fig_month_compare = px.line(
            month_compare,
            x="Mês nome",
            y="Registros",
            color="Base",
            markers=True,
            category_orders={"Mês nome": list(MESES.values())},
            color_discrete_map={"LESFEM": PURPLE, "SENASP": SAGE},
            title="Evolução mensal · total de registros",
            labels={"Mês nome": "Mês"},
        )
        fig_month_compare.update_traces(line=dict(width=3), marker=dict(size=7))
        style_fig(fig_month_compare, 420)
        st.plotly_chart(fig_month_compare, use_container_width=True, config={"displayModeBar": False})

    # Divergência municipal: usamos as colunas consolidadas apenas nesta comparação.
    div = base.loc[base["Diferença LESFEM − SENASP"] != 0, [
        "Município",
        "LESFEM - Total de registros 2025",
        "SENASP - Total vítimas feminicídio/tentativa 2025",
        "Diferença LESFEM − SENASP",
    ]].copy()
    div["Abs"] = div["Diferença LESFEM − SENASP"].abs()
    div = div.nlargest(14, "Abs").sort_values("Abs")
    div["Sentido"] = np.where(div["Diferença LESFEM − SENASP"] >= 0, "LESFEM maior", "SENASP maior")
    fig_div = px.bar(
        div,
        x="Diferença LESFEM − SENASP",
        y="Município",
        orientation="h",
        color="Sentido",
        color_discrete_map={"LESFEM maior": PURPLE, "SENASP maior": SAGE},
        title="Maiores diferenças municipais entre LESFEM e SENASP",
        hover_data=["LESFEM - Total de registros 2025", "SENASP - Total vítimas feminicídio/tentativa 2025"],
    )
    style_fig(fig_div, 520)
    fig_div.update_layout(yaxis_title="", xaxis_title="LESFEM − SENASP")
    st.plotly_chart(fig_div, use_container_width=True, config={"displayModeBar": False})

# ------------------------------------------------------------
# 4) ESTUPROS
# ------------------------------------------------------------
with tabs[3]:
    st.subheader("Estupros contra mulheres no Paraná em 2025")

    monthly_rape = (
        estupro.groupby("mes_num", as_index=False)["feminino"]
        .sum()
        .sort_values("mes_num")
    )
    monthly_rape["Mês"] = monthly_rape["mes_num"].map(MESES)
    monthly_rape["Acumulado"] = monthly_rape["feminino"].cumsum()

    max_row = monthly_rape.loc[monthly_rape["feminino"].idxmax()]
    min_row = monthly_rape.loc[monthly_rape["feminino"].idxmin()]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Total em 2025", br_int(estupro_total), "vítimas mulheres")
    with c2:
        kpi("Média mensal", br_num(estupro_total / max(len(monthly_rape), 1), 1), "média dos meses de 2025")
    with c3:
        kpi("Maior mês", f"{max_row['Mês']} · {br_int(max_row['feminino'])}", "maior valor mensal")
    with c4:
        kpi("Menor mês", f"{min_row['Mês']} · {br_int(min_row['feminino'])}", "menor valor mensal")

    

    fig_rape = go.Figure()
    fig_rape.add_trace(go.Bar(
        x=monthly_rape["Mês"],
        y=monthly_rape["feminino"],
        name="Vítimas no mês",
        marker_color=PURPLE,
        hovertemplate="%{x}<br>Vítimas: %{y}<extra></extra>",
    ))
    fig_rape.add_trace(go.Scatter(
        x=monthly_rape["Mês"],
        y=monthly_rape["Acumulado"],
        name="Acumulado",
        mode="lines+markers",
        line=dict(color=SAGE, width=3),
        marker=dict(size=7),
        yaxis="y2",
        hovertemplate="%{x}<br>Acumulado: %{y}<extra></extra>",
    ))
    fig_rape.update_layout(
        
    )
    style_fig(fig_rape, 500)
    st.plotly_chart(fig_rape, use_container_width=True, config={"displayModeBar": False})

    st.markdown("#### Tabela mensal")
    rape_table = monthly_rape[["Mês", "feminino", "Acumulado"]].rename(columns={"feminino": "Vítimas mulheres"})
    st.dataframe(rape_table, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="method-card">
            <strong>Leitura responsável:</strong> o gráfico mostra o que consta na base utilizada para a variável
            feminina. Ele não mede subnotificação e, por ser uma base agregada, não permite identificar vítimas distintas
            ao longo do ano sem uma chave individual.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# 5) TERRITÓRIO E REDE — SEM LESFEM
# ------------------------------------------------------------
with tabs[4]:
    st.subheader("Território, rede de atendimento e contexto municipal")
    terr = apply_territorial_filters(base, "territorio", show_population=False)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Municípios no recorte", br_int(len(terr)), "após os filtros aplicados")
    with c2:
        kpi("Com Delegacia da Mulher", br_int((terr["Possui Delegacia da Mulher?"].astype(str).str.casefold() == "sim").sum()), "base consolidada")
    with c3:
        kpi("Com CAPS", br_int((terr["Possui CAPS?"].astype(str).str.casefold() == "sim").sum()), "base consolidada")
    with c4:
        kpi("Quantidade de CAPS", br_int(terr["Quantidade de CAPS"].sum()), "soma no recorte")

    c_left, c_right = st.columns(2)
    with c_left:
        reg = (
            terr.groupby("Macrorregional de Saúde", as_index=False)
            .agg(
                Municipios=("Município", "count"),
                **{"Feminicídio + tentativa": ("Feminicídio + tentativa 2025", "sum")},
                CAPS=("Quantidade de CAPS", "sum"),
                Populacao=(POPULATION_COL, "sum"),
            )
        )
        reg["Feminicídio + tentativa por 100 mil"] = safe_rate(
            reg["Feminicídio + tentativa"], reg["Populacao"]
        )
        fig_reg = px.bar(
            reg,
            x="Macrorregional de Saúde",
            y="Feminicídio + tentativa por 100 mil",
            color="Feminicídio + tentativa por 100 mil",
            color_continuous_scale=[GREEN, SAGE, LAVENDER, PURPLE, MAGENTA],
            title="Feminicídio + tentativa por 100 mil · macrorregional",
            hover_data=["Feminicídio + tentativa", "Populacao", "CAPS", "Municipios"],
        )
        style_fig(fig_reg, 430)
        fig_reg.update_layout(xaxis_title="", xaxis_tickangle=-18, coloraxis_showscale=False)
        st.plotly_chart(fig_reg, use_container_width=True, config={"displayModeBar": False})

    with c_right:
        scatter_df = terr.dropna(subset=[POPULATION_COL]).copy()
        fig_sc = px.scatter(
            scatter_df,
            x=POPULATION_COL,
            y="Feminicídio + tentativa por 100 mil",
            size="Quantidade de CAPS",
            color="Possui Delegacia da Mulher?",
            hover_name="Município",
            hover_data=["Regional de Saúde", "Feminicídio + tentativa 2025"],
            color_discrete_map={"Sim": PURPLE, "Não": GREEN, "sim": PURPLE, "não": GREEN, "Nao": GREEN},
            title="População × feminicídio/tentativa por 100 mil · bolha = CAPS",
            labels={"Feminicídio + tentativa por 100 mil": "Feminicídio + tentativa por 100 mil"},
        )
        style_fig(fig_sc, 430)
        st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        """
        <div class="section-note">
            A presença de CAPS ou Delegacia da Mulher é mostrada como <strong>contexto territorial</strong> e não entra
            automaticamente no score de violência. Mais equipamentos públicos não significam maior violência; da mesma
            forma, associações visuais não demonstram relação causal.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# 6) CONTEXTO IBGE — RELAÇÕES ENTRE CONTEXTO E VIOLÊNCIA
# ------------------------------------------------------------
with tabs[5]:
    st.subheader("Contexto socioeconômico e violência · IBGE")
    app_caption(
        "Esta aba cruza indicadores demográficos e socioeconômicos do IBGE com indicadores de violência contra a mulher. "
        "A população e a densidade demográfica têm referência no Censo Demográfico 2022; os demais indicadores mantêm "
        "o ano indicado no próprio nome. As associações são exploratórias e não demonstram causalidade."
    )

    ctx = apply_territorial_filters(scored, "ibge_contexto", show_population=True)

    if ctx.empty:
        st.warning("Nenhum município corresponde aos filtros selecionados.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi(
                "População Censo 2022",
                br_int(ctx[POPULATION_COL].sum()),
                "soma municipal · IBGE",
            )
        with c2:
            kpi(
                "PIB per capita mediano",
                f"R$ {br_num(ctx['IBGE - PIB per capita - R$ [2023]'].median(), 0)}",
                "IBGE · 2023",
            )
        with c3:
            kpi(
                "IDHM mediano",
                br_num(ctx["IBGE - IDHM [2010]"].median(), 3),
                "IBGE · 2010",
            )
        with c4:
            kpi(
                "Densidade mediana",
                br_num(ctx["IBGE - Densidade demográfica - hab/km² [2022]"].median(), 1),
                "hab./km² · Censo 2022",
            )

        st.markdown(
            """
            <div class="section-note">
                <strong>O que esta aba procura responder?</strong> Por exemplo: municípios com PIB per capita mais baixo
                apresentam maiores taxas de feminicídio, tentativa, lesão corporal ou ameaça? Municípios com IDHM mais baixo
                apresentam valores maiores ou menores desses indicadores? O painel mede <strong>associação estatística</strong>,
                não causa e efeito. Também é importante observar que os anos de referência dos indicadores socioeconômicos
                não são todos os mesmos dos registros de violência de 2025.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Indicadores IBGE disponíveis para cruzamento.
        context_metrics = {
            "População · Censo 2022": POPULATION_COL,
            "Área territorial (km²) · 2025": "IBGE - Área Territorial - km² [2025]",
            "Densidade demográfica (hab./km²) · 2022": "IBGE - Densidade demográfica - hab/km² [2022]",
            "Escolarização de 6 a 14 anos (%) · 2022": "IBGE - Escolarização 6 a 14 anos - % [2022]",
            "IDHM · 2010": "IBGE - IDHM [2010]",
            "Mortalidade infantil · 2025": "IBGE - Mortalidade infantil - óbitos por mil nascidos vivos [2025]",
            "PIB per capita (R$) · 2023": "IBGE - PIB per capita - R$ [2023]",
            "Receita 2025 por habitante (R$) · base Censo 2022": "IBGE - Receita por habitante - R$",
            "Despesa 2025 por habitante (R$) · base Censo 2022": "IBGE - Despesa por habitante - R$",
            "Saldo fiscal 2025 por habitante (R$) · base Censo 2022": "IBGE - Saldo fiscal por habitante - R$",
        }

        # Aqui o foco é a violência propriamente dita, e não apenas o score composto.
        violence_metrics = {
            "Lesão corporal · taxa por 100 mil": "Lesão corporal por 100 mil",
            "Ameaça · taxa por 100 mil": "Ameaça por 100 mil",
            "Feminicídio consumado · por 100 mil": "Feminicídios por 100 mil",
            "Tentativa de feminicídio · por 100 mil": "Tentativas de feminicídio por 100 mil",
            "Feminicídio + tentativa · por 100 mil": "Feminicídio + tentativa por 100 mil",
        }

        # ----------------------------------------------------
        # CRUZAMENTO LIVRE: UM INDICADOR IBGE × UMA VIOLÊNCIA
        # ----------------------------------------------------
        st.markdown("### Cruzar indicador do IBGE com indicador de violência")
        sx1, sx2, sx3 = st.columns([1, 1, .8])
        with sx1:
            context_label = st.selectbox(
                "Indicador do IBGE",
                options=list(context_metrics.keys()),
                index=list(context_metrics.keys()).index("PIB per capita (R$) · 2023"),
                key="ibge_context_metric",
            )
        with sx2:
            violence_label = st.selectbox(
                "Indicador de violência",
                options=list(violence_metrics.keys()),
                index=2,
                key="ibge_violence_metric",
            )
        with sx3:
            context_color = st.selectbox(
                "Cor por",
                ["Macrorregional de Saúde", "Possui Delegacia da Mulher?", "Possui CAPS?"],
                key="ibge_context_color",
            )

        context_col = context_metrics[context_label]
        violence_col = violence_metrics[violence_label]

        plot_ctx = (
            ctx.replace([np.inf, -np.inf], np.nan)
            .dropna(subset=[context_col, violence_col, POPULATION_COL])
            .copy()
        )

        if plot_ctx.empty:
            st.warning("Não há municípios com dados suficientes para esse cruzamento.")
        else:
            # Spearman sem scipy: Pearson aplicado aos ranks.
            if plot_ctx[context_col].nunique() > 1 and plot_ctx[violence_col].nunique() > 1:
                rho_selected = plot_ctx[context_col].rank(method="average").corr(
                    plot_ctx[violence_col].rank(method="average")
                )
            else:
                rho_selected = np.nan

            r1, r2, r3 = st.columns(3)
            with r1:
                kpi(
                    "Correlação de Spearman",
                    br_num(rho_selected, 3),
                    "−1 a +1 · associação monotônica",
                )
            with r2:
                kpi(
                    "Municípios analisados",
                    br_int(len(plot_ctx)),
                    "com dados nas duas variáveis",
                )
            with r3:
                if pd.isna(rho_selected):
                    interpretation = "dados insuficientes"
                elif abs(rho_selected) < .20:
                    interpretation = "muito fraca"
                elif abs(rho_selected) < .40:
                    interpretation = "fraca"
                elif abs(rho_selected) < .60:
                    interpretation = "moderada"
                elif abs(rho_selected) < .80:
                    interpretation = "forte"
                else:
                    interpretation = "muito forte"
                direction = "positiva" if pd.notna(rho_selected) and rho_selected > 0 else "negativa" if pd.notna(rho_selected) and rho_selected < 0 else "sem direção"
                kpi("Leitura descritiva", interpretation, direction)

            fig_ctx = px.scatter(
                plot_ctx,
                x=context_col,
                y=violence_col,
                size=POPULATION_COL,
                size_max=34,
                color=context_color,
                color_discrete_sequence=PALETTE,
                hover_name="Município",
                hover_data={
                    POPULATION_COL: ":,.0f",
                    "Lesão corporal por 100 mil": ":.2f",
                    "Ameaça por 100 mil": ":.2f",
                    "Feminicídios por 100 mil": ":.2f",
                    "Tentativas de feminicídio por 100 mil": ":.2f",
                    "IBGE - PIB per capita - R$ [2023]": ":,.2f",
                    "IBGE - IDHM [2010]": ":.3f",
                },
                title=f"{context_label} × {violence_label}",
                labels={context_col: context_label, violence_col: violence_label},
            )
            style_fig(fig_ctx, 540)
            st.plotly_chart(fig_ctx, use_container_width=True, config={"displayModeBar": False})

            # ------------------------------------------------
            # COMPARAÇÃO POR QUARTIS: BAIXO × ALTO CONTEXTO
            # ------------------------------------------------
            st.markdown("### Comparar municípios com valores baixos e altos do indicador IBGE")
            quart_df = plot_ctx[["Município", context_col, violence_col]].dropna().copy()
            if len(quart_df) >= 8 and quart_df[context_col].nunique() >= 4:
                # Usa o rank apenas para formar quatro grupos com tamanhos semelhantes,
                # preservando o valor original para hover e leitura.
                quart_df["Faixa do indicador IBGE"] = pd.qcut(
                    quart_df[context_col].rank(method="first"),
                    q=4,
                    labels=[
                        "Q1 · valores mais baixos",
                        "Q2 · baixo-intermediário",
                        "Q3 · alto-intermediário",
                        "Q4 · valores mais altos",
                    ],
                )
                quart_summary = (
                    quart_df.groupby("Faixa do indicador IBGE", observed=True)
                    .agg(
                        **{
                            "Mediana da violência": (violence_col, "median"),
                            "Média da violência": (violence_col, "mean"),
                            "Municípios": ("Município", "count"),
                            "Mediana do indicador IBGE": (context_col, "median"),
                        }
                    )
                    .reset_index()
                )

                fig_quart = px.bar(
                    quart_summary,
                    x="Faixa do indicador IBGE",
                    y="Mediana da violência",
                    color="Faixa do indicador IBGE",
                    color_discrete_sequence=PALETTE,
                    hover_data={
                        "Média da violência": ":.2f",
                        "Municípios": True,
                        "Mediana do indicador IBGE": ":.2f",
                    },
                    title=f"Mediana de {violence_label.lower()} por quartil de {context_label.lower()}",
                    labels={"Mediana da violência": violence_label},
                )
                style_fig(fig_quart, 450)
                fig_quart.update_layout(showlegend=False, xaxis_title="")
                st.plotly_chart(fig_quart, use_container_width=True, config={"displayModeBar": False})

                q1 = quart_summary.iloc[0]
                q4 = quart_summary.iloc[-1]
                app_caption(
                    f"Leitura direta: no quartil com valores mais baixos de {context_label}, a mediana de {violence_label} "
                    f"é {br_num(q1['Mediana da violência'], 2)}; no quartil com valores mais altos, "
                    f"é {br_num(q4['Mediana da violência'], 2)}. Essa comparação é descritiva e não controla outros fatores."
                )
            else:
                app_caption("Não há dados suficientes para dividir esse indicador em quatro grupos comparáveis.")

        # ----------------------------------------------------
        # MATRIZ: TODOS OS INDICADORES IBGE × TODAS AS VIOLÊNCIAS
        # ----------------------------------------------------
        st.markdown("### Matriz de correlações · IBGE × violência")
        app_caption(
            "Cada célula mostra a correlação de Spearman entre um indicador do IBGE e um indicador de violência. "
            "Assim é possível ver, por exemplo, se PIB per capita e IDHM apresentam associação positiva, negativa ou próxima de zero "
            "com lesão corporal, ameaça, feminicídio e tentativa."
        )

        matrix_values = []
        matrix_n = []
        context_labels = list(context_metrics.keys())
        violence_labels = list(violence_metrics.keys())

        for v_label in violence_labels:
            v_col = violence_metrics[v_label]
            row_values = []
            row_n = []
            for c_label in context_labels:
                c_col = context_metrics[c_label]
                pair = ctx[[c_col, v_col]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(pair) >= 3 and pair[c_col].nunique() > 1 and pair[v_col].nunique() > 1:
                    rho = pair[c_col].rank(method="average").corr(
                        pair[v_col].rank(method="average")
                    )
                else:
                    rho = np.nan
                row_values.append(rho)
                row_n.append(len(pair))
            matrix_values.append(row_values)
            matrix_n.append(row_n)

        heat_text = []
        for i, row in enumerate(matrix_values):
            heat_text.append([
                "—" if pd.isna(value) else f"{value:.2f}"
                for value in row
            ])

        fig_heat = go.Figure(
            data=go.Heatmap(
                z=matrix_values,
                x=context_labels,
                y=violence_labels,
                zmin=-1,
                zmax=1,
                zmid=0,
                colorscale=[
                    [0.00, GREEN],
                    [0.25, SAGE],
                    [0.50, BG_2],
                    [0.75, LAVENDER],
                    [1.00, MAGENTA],
                ],
                text=heat_text,
                customdata=matrix_n,
                hovertemplate=(
                    "<b>%{y}</b><br>%{x}<br>Spearman: %{z:.3f}"
                    "<br>Municípios com dados: %{customdata}<extra></extra>"
                ),
                colorbar=dict(
                    title=dict(text="Spearman", font=dict(color=MUTED)),
                    tickfont=dict(color=MUTED),
                    thickness=12,
                ),
            )
        )
        fig_heat.update_layout(title="Associação entre indicadores socioeconômicos e violência")
        style_fig(fig_heat, 620)
        fig_heat.update_xaxes(tickangle=-35)
        fig_heat.update_yaxes(automargin=True)
        st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

        # Tabela ordenável facilita identificar associações mais fortes sem depender só da cor.
        correlation_rows = []
        for v_idx, v_label in enumerate(violence_labels):
            for c_idx, c_label in enumerate(context_labels):
                rho = matrix_values[v_idx][c_idx]
                correlation_rows.append({
                    "Indicador IBGE": c_label,
                    "Indicador de violência": v_label,
                    "Spearman": rho,
                    "Municípios com dados": matrix_n[v_idx][c_idx],
                    "Força absoluta": abs(rho) if pd.notna(rho) else np.nan,
                })
        correlation_df = pd.DataFrame(correlation_rows).sort_values(
            ["Força absoluta", "Municípios com dados"],
            ascending=[False, False],
            na_position="last",
        )
        st.dataframe(
            correlation_df.drop(columns="Força absoluta").style.format({"Spearman": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
            height=430,
        )

        st.markdown(
            """
            <div class="section-note">
                <strong>Como interpretar:</strong> Spearman próximo de +1 indica que valores maiores do indicador IBGE
                tendem a aparecer junto de valores maiores do indicador de violência; próximo de −1 indica que valores maiores
                de um tendem a aparecer com valores menores do outro; próximo de 0 indica pouca associação monotônica.
                <strong>Correlação não implica causalidade.</strong> Além disso, feminicídios são eventos relativamente raros,
                especialmente em municípios pequenos, então suas taxas podem oscilar bastante com uma única ocorrência.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Mantemos a leitura por porte populacional como análise complementar,
        # mas ela deixa de ser o foco principal da aba.
        with st.expander("Análise complementar · violência por porte populacional", expanded=False):
            population_bins = [-np.inf, 10_000, 25_000, 50_000, 100_000, 500_000, np.inf]
            population_labels = [
                "Até 10 mil",
                "10 a 25 mil",
                "25 a 50 mil",
                "50 a 100 mil",
                "100 a 500 mil",
                "Mais de 500 mil",
            ]
            size_df = ctx.dropna(subset=[POPULATION_COL]).copy()
            size_df["Porte populacional"] = pd.cut(
                size_df[POPULATION_COL],
                bins=population_bins,
                labels=population_labels,
                include_lowest=True,
            )
            size_metric_label = st.selectbox(
                "Indicador de violência para comparar por porte",
                options=violence_labels,
                key="ibge_population_size_violence",
            )
            size_metric_col = violence_metrics[size_metric_label]
            size_plot = size_df.dropna(subset=[size_metric_col]).copy()
            fig_size = px.box(
                size_plot,
                x="Porte populacional",
                y=size_metric_col,
                points="outliers",
                color="Porte populacional",
                color_discrete_sequence=PALETTE,
                title=f"{size_metric_label} segundo o porte populacional",
                labels={size_metric_col: size_metric_label},
            )
            style_fig(fig_size, 470)
            fig_size.update_layout(showlegend=False, xaxis_title="")
            st.plotly_chart(fig_size, use_container_width=True, config={"displayModeBar": False})

# ------------------------------------------------------------
# 7) BASES COMPLETAS
# ------------------------------------------------------------
with tabs[6]:
    st.subheader("Consulta às bases na íntegra")
    dataset = st.selectbox(
        "Base",
        [
            "Base municipal consolidada",
            "LESFEM · base detalhada caso a caso",
            "Feminicídio e tentativa · mensal por município",
            "Estupro · mensal estadual · mulheres",
        ],
    )

    if dataset == "Base municipal consolidada":
        table = publicize_columns(base_raw.copy())
        app_caption("Base municipal integrada com os campos demográficos e socioeconômicos do IBGE.")
        q = st.text_input("Buscar município", placeholder="Ex.: Curitiba")
        if q:
            table = table[table["Município"].astype(str).str.contains(q, case=False, na=False)]
        filename = "base_municipal_2025_com_ibge.csv"
    elif dataset.startswith("LESFEM"):
        table = lesfem_raw.copy()
        c1, c2, c3 = st.columns(3)
        with c1:
            q = st.text_input("Buscar município", placeholder="Ex.: Cascavel", key="search_lesfem")
        with c2:
            les_event = st.selectbox(
                "Desfecho", ["Todos"] + sorted(table["Consumado ou Tentado"].dropna().astype(str).unique().tolist()),
                key="table_les_event"
            )
        with c3:
            les_source = st.selectbox(
                "Origem", ["Todos"] + sorted(table["Fonte"].dropna().astype(str).unique().tolist()),
                key="table_les_source"
            )
        if q:
            table = table[table["Município"].astype(str).str.contains(q, case=False, na=False)]
        if les_event != "Todos":
            table = table[table["Consumado ou Tentado"] == les_event]
        if les_source != "Todos":
            table = table[table["Fonte"] == les_source]
        filename = "lesfem_2025.csv"
    elif dataset.startswith("Feminicídio e tentativa"):
        table = fem_raw.copy()
        c1, c2 = st.columns(2)
        with c1:
            q = st.text_input("Buscar município", placeholder="Ex.: Londrina", key="search_fem")
        with c2:
            evento = st.selectbox("Evento", ["Todos"] + sorted(table["evento"].dropna().unique().tolist()))
        if q:
            table = table[table["municipio"].astype(str).str.contains(q, case=False, na=False)]
        if evento != "Todos":
            table = table[table["evento"] == evento]
        filename = "feminicidio_tentativa_2025.csv"
    else:
        table = estupro_raw.copy()
        filename = "estupro_mulheres_2025.csv"

    app_caption(f"{br_int(len(table))} linhas exibidas")
    st.dataframe(table, use_container_width=True, height=610, hide_index=True)
    st.download_button(
        "Baixar CSV exibido",
        data=csv_download_bytes(table),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )

# ------------------------------------------------------------
# 8) METODOLOGIA E FONTES
# ------------------------------------------------------------
with tabs[7]:
    st.subheader("Metodologia, score e fontes")

    st.markdown(
        """
        <div class="method-card">
            <strong>Separação das fontes:</strong><br>
            O painel usa uma base consolidada para score, ranking, mapa e análises
            municipais gerais. A base do LESFEM é explorada em uma aba própria, com gráficos descritivos,
            perfil dos registros, ranking municipal interno da fonte e uma comparação secundária com o SENASP.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Como o score é calculado")
    st.markdown(
        """
        1. O usuário escolhe quais indicadores de violência deseja combinar.
        2. Todos entram no score como **taxas por 100 mil habitantes**, permitindo comparação proporcional entre municípios de tamanhos diferentes.
        3. Para **lesão corporal e ameaça**, a base consolidada já fornece as taxas. Como a nova base não contém as contagens absolutas desses dois indicadores, o painel preserva as taxas informadas pela fonte e não inventa contagens.
        4. Para **feminicídio consumado e tentativa (SENASP)**, o painel parte das contagens absolutas e calcula `casos / população de referência × 100.000`.
        5. A população de referência é a **população municipal do Censo Demográfico IBGE 2022**, último censo disponível. Como os registros de violência analisados são de 2025, o painel informa explicitamente que o denominador populacional tem referência em 2022.
        6. Cada taxa é transformada em uma **posição percentual entre os municípios com dado válido**. Quanto maior o valor relativo, maior o componente do score.
        7. O score final é sempre uma **média ponderada**: `Score = Σ(posição relativa × peso) / Σ(pesos)`.
        8. Pesos padrão: **ameaça = 1**, **lesão corporal = 2**, **tentativa de feminicídio = 4** e **feminicídio consumado = 5**. Eles são uma escolha analítica do painel, editável pelo usuário, e não pesos oficiais das fontes.
        9. O score varia de **0 a 100** e deve ser lido como posição relativa dentro do Paraná, não como probabilidade individual, índice oficial ou prova causal.
        """
    )

    st.markdown(
        """
        <div class="section-note">
            <strong>Municípios pequenos:</strong> taxas proporcionais são importantes para comparar populações de tamanhos
            diferentes, mas uma única ocorrência pode elevar muito a taxa de um município pequeno. Por isso o painel
            mostra a população e oferece filtro de população mínima sem alterar a base estadual usada para calcular os percentis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    methodology = pd.DataFrame([
        ["IBGE", "População e densidade do Censo Demográfico 2022; demais indicadores mantêm o ano de referência indicado no próprio campo", "Município", "Censo 2022 como denominador populacional das taxas derivadas do SENASP e demais variáveis como contexto demográfico/socioeconômico"],
        ["SESP-PR / CAPE / base consolidada", "Lesão corporal e ameaça — taxas por 100 mil informadas na base", "Município", "Indicadores elegíveis para o score"],
        ["SENASP / Sinesp VDE", "Feminicídios consumados e tentativas", "Município e mês", "Indicadores elegíveis para o score e séries temporais"],
        ["LESFEM / UEL", "Feminicídios consumados e tentativas + contexto caso a caso", "Caso / município", "Análise descritiva própria; comparação secundária com SENASP"],
        ["SENASP / Sinesp VDE", "Estupro — vítimas mulheres", "Estado e mês", "Aba estadual; não entra no score municipal"],
        ["Base consolidada", "Delegacia da Mulher, CAPS e Regionais de Saúde", "Município", "Filtros e contexto territorial; não entram automaticamente no score"],
        ["IBGE — Malhas", "Limites municipais", "Município", "Geometria do mapa; não é fonte dos indicadores de violência"],
    ], columns=["Fonte", "Dado", "Granularidade", "Uso no painel"])
    st.dataframe(methodology, use_container_width=True, hide_index=True)

    st.markdown("#### Sobre a divergência LESFEM × SENASP")
    st.markdown(
        f"""
        Nos arquivos carregados, o LESFEM totaliza **{br_int(lesfem_total)}** feminicídios consumados ou tentados,
        enquanto o SENASP/Sinesp VDE totaliza **{br_int(senasp_total)}**. A diferença é de **{br_int(diff_total)} registros**
        (**{br_num(diff_pct, 1)}%** em relação ao total do SENASP). Como as metodologias e os processos de consolidação não
        são idênticos, o painel não mistura essas bases no score. A divergência é exibida separadamente para consulta.
        """
    )

    st.markdown("#### Regras de leitura")
    st.markdown(
        """
        - O score serve para **comparação relativa entre municípios**, de acordo com os indicadores e pesos que o usuário escolher.
        - Um score alto significa que o município está em posições altas nos indicadores selecionados; não significa, sozinho, que toda forma de violência é maior naquele território.
        - CAPS e Delegacia da Mulher são variáveis de contexto e **não são tratadas como violência**.
        - LESFEM e SENASP **não são somados**.
        - Estupro não é municipalizado porque o arquivo disponibilizado para esta análise não contém município.
        - Os indicadores IBGE são usados para **contextualização e análise exploratória** e não entram automaticamente no score de violência.
        - Correlações entre score e IDHM, PIB, densidade, escolarização ou outros indicadores **não demonstram causalidade**.
        - A ausência de registro em uma base agregada não deve ser interpretada como prova de ausência do fenômeno para além do que a própria base mede.
        """
    )

    st.markdown("#### Fontes")
    st.markdown(
        f"""
        - [IPARDES — Base de Dados do Estado]({SOURCES['IPARDES']})
        - [Ministério da Justiça e Segurança Pública — SENASP / Sinesp VDE]({SOURCES['SENASP']})
        - [LESFEM/UEL — Base de dados e nota metodológica]({SOURCES['LESFEM']})
        - [Secretaria da Segurança Pública do Paraná — CAPE]({SOURCES['CAPE']})
        - [IBGE — Cidades e Estados]({SOURCES['IBGE_CIDADES']})
        - [IBGE — Malhas territoriais]({SOURCES['IBGE_MALHAS']})
        """
    )

# Rodapé
st.markdown("---")
st.markdown(
    f"""
    <div class="small-muted">
        Painel 2025 · Paraná 
    </div>
    """,
    unsafe_allow_html=True,
)
