"""
Leitura, limpeza e criação de variáveis da base de retenção.

Tudo que é usado tanto no notebook quanto no app fica aqui, para que as duas
pontas trabalhem exatamente com a mesma definição de cada variável.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ARQ_EXCEL = RAIZ / "bd_case_final_1.xlsx"
ARQ_CACHE = RAIZ / "data" / "base_bruta.parquet"

PRODUTOS = ["BL", "TV", "FONE", "MV"]

# O móvel usa nomes de coluna diferentes dos produtos residenciais
COL_ID = {
    "BL": "ID_PRODUTO_BL_{m}",
    "TV": "ID_PRODUTO_TV_{m}",
    "FONE": "ID_PRODUTO_FONE_{m}",
    "MV": "COD_PLANO_MOVEL_{m}",
}
COL_DSC = {
    "BL": "DSC_PRODUTO_BL_{m}",
    "TV": "DSC_PRODUTO_TV_{m}",
    "FONE": "DSC_PRODUTO_FONE_{m}",
    "MV": "DSC_PLANO_MOVEL_{m}",
}


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------
def carregar_base(usar_cache: bool = True) -> pd.DataFrame:
    """Lê o Excel original. Na primeira leitura salva um parquet, porque o
    xlsx de 80 MB leva alguns minutos para abrir e o parquet, segundos."""
    if usar_cache and ARQ_CACHE.exists():
        return pd.read_parquet(ARQ_CACHE)
    df = pd.read_excel(ARQ_EXCEL, engine="openpyxl")
    ARQ_CACHE.parent.mkdir(exist_ok=True)
    df.to_parquet(ARQ_CACHE, index=False)
    return df


# ---------------------------------------------------------------------------
# Limpeza
# ---------------------------------------------------------------------------
def limpar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # O case fala em jun/25 a dez/25; há 2 leads perdidos em fev e mai
    df = df[df["DATA_FUNIL_SOLAR"] >= "2025-06-01"].reset_index(drop=True)

    # Vazio em MPLAY não é dado faltante: pelo dicionário significa "não elegível"
    for m in ["M0", "M30"]:
        df[f"DSC_CLI_MPLAY_{m}"] = df[f"DSC_CLI_MPLAY_{m}"].fillna("NAO ELEGIVEL")

    # 38 cidades vieram com classificação "-", mas o grupo de preço é MD2
    df.loc[df["CLASSIFICACAO"] == "-", "CLASSIFICACAO"] = "MD2"

    # Corrige o "PADRÃO" que vem com encoding quebrado em alguns ambientes
    df["GRUPO_DA_TABELA_DE_PRECOS"] = (
        df["GRUPO_DA_TABELA_DE_PRECOS"].astype(str).str.replace("PADR.O", "PADRAO", regex=True)
    )

    for p in PRODUTOS:
        for m in ["M0", "M30"]:
            df[COL_DSC[p].format(m=m)] = df[COL_DSC[p].format(m=m)].fillna("").str.upper().str.strip()

    df["MEIO_FISICO_BROWNFIELD"] = df["MEIO_FISICO_BROWNFIELD"].fillna("NAO SE APLICA")

    # Valores negativos (253 leads) parecem créditos/estornos lançados no
    # produto. São poucos; mantemos, mas sinalizamos.
    cols_valor = [c for c in df.columns if c.startswith("VALOR_")]
    df["flag_valor_negativo"] = (df[cols_valor] < 0).any(axis=1).astype(int)

    df = df.sort_values(["id_cliente", "DATA_FUNIL_SOLAR"]).reset_index(drop=True)
    df.insert(0, "id_lead", np.arange(len(df)))
    return df


# ---------------------------------------------------------------------------
# Variáveis de M0 (o que se sabe do cliente quando ele entra no funil)
# ---------------------------------------------------------------------------
def _velocidade_mbps(desc: pd.Series) -> pd.Series:
    """Extrai a velocidade da banda larga a partir da descrição comercial.
    Ex.: 'BL 350M GPLAY...' -> 350 ; 'BL 1GIGA ...' -> 1000"""
    ext = desc.str.extract(r"(\d+)\s?(MEGA|GIGA|M|G)\b")
    vel = pd.to_numeric(ext[0], errors="coerce")
    giga = ext[1].isin(["G", "GIGA"])
    vel = np.where(giga, vel * 1000, vel)
    return pd.Series(vel, index=desc.index)


def _faixa_velocidade(vel: pd.Series) -> pd.Series:
    bins = [0, 299, 399, 549, 799, np.inf]
    rot = ["ate 250M", "300-350M", "400-500M", "600-750M", "1G ou mais"]
    return pd.cut(vel, bins=bins, labels=rot).astype(str).replace("nan", "sem BL / suspensa")


def classificar_tv(desc: pd.Series) -> pd.Series:
    cond = [
        desc.eq(""),
        desc.str.contains("4K"),
        desc.str.contains("TOP"),
        desc.str.contains("MIX"),
        desc.str.contains("INICIAL|FACIL"),
    ]
    rot = ["sem TV", "TOP 4K", "TOP", "MIX", "ENTRADA"]
    return pd.Series(np.select(cond, rot, "OUTROS"), index=desc.index)


def _tipo_movel(desc: pd.Series) -> pd.Series:
    cond = [desc.eq(""), desc.str.contains("PÓS|POS"), desc.str.contains("CONTROLE")]
    return pd.Series(np.select(cond, ["sem plano", "POS", "CONTROLE"], "OUTRO"), index=desc.index)


def criar_variaveis_m0(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for p in PRODUTOS:
        df[f"tem_{p.lower()}"] = (df[COL_ID[p].format(m="M0")] > 0).astype(int)

    df["qtd_produtos"] = df[[f"tem_{p.lower()}" for p in PRODUTOS]].sum(axis=1)
    df["qtd_produtos_res"] = df[["tem_bl", "tem_tv", "tem_fone"]].sum(axis=1)

    def _combo(r):
        partes = [p for p in PRODUTOS if r[f"tem_{p.lower()}"]]
        return "+".join(partes) if partes else "NENHUM"

    df["combo_m0"] = df.apply(_combo, axis=1)

    df["receita_m0"] = df[[f"VALOR_{p}_M0" for p in PRODUTOS]].sum(axis=1)

    # Banda larga
    dsc_bl = df["DSC_PRODUTO_BL_M0"]
    df["bl_suspensa_m0"] = dsc_bl.str.contains("SUSPENS").astype(int)
    df["velocidade_bl_m0"] = _velocidade_mbps(dsc_bl).where(df["bl_suspensa_m0"] == 0)
    df["faixa_velocidade_bl"] = _faixa_velocidade(df["velocidade_bl_m0"])
    df["bl_gplay"] = dsc_bl.str.contains("GPLAY").astype(int)
    df["bl_pon"] = dsc_bl.str.contains(r"\bPON\b|FIBRA").astype(int)

    # TV
    dsc_tv = df["DSC_PRODUTO_TV_M0"]
    df["tier_tv"] = classificar_tv(dsc_tv)
    df["tv_streaming"] = dsc_tv.str.contains("STREAMING|CTV\\+|TVMAIS").astype(int)

    # Móvel
    df["tipo_movel"] = _tipo_movel(df["DSC_PLANO_MOVEL_M0"])

    # Marcas de "histórico" que ficam na própria descrição do plano:
    # RET = plano de retenção (cliente já passou por retenção antes)
    # OLD = plano de catálogo antigo / descontinuado
    # MIG = plano de migração
    dsc_res = dsc_bl + " | " + dsc_tv
    df["plano_de_retencao_m0"] = dsc_res.str.contains(r"\bRET\b").astype(int)
    df["plano_old_m0"] = dsc_res.str.contains(r"\bOLD\b").astype(int)
    df["plano_migracao_m0"] = dsc_res.str.contains(r"\bMIG").astype(int)

    # Participação de cada produto no ticket
    df["peso_tv_no_ticket"] = np.where(df["receita_m0"] > 0, df["VALOR_TV_M0"] / df["receita_m0"], 0)

    # Sobrepreço: quanto o cliente paga acima (ou abaixo) do preço típico de
    # um produto equivalente. Para BL, equivalente = mesma faixa de velocidade
    # na mesma classificação de praça; para TV, mesmo nível de pacote.
    # A ideia: quem paga caro tem "espaço" para receber uma oferta de troca.
    tem_valor_bl = df["VALOR_BL_M0"] > 0
    ref_bl = (df[tem_valor_bl].groupby(["faixa_velocidade_bl", "CLASSIFICACAO"])["VALOR_BL_M0"]
              .median().rename("ref_bl"))
    df = df.join(ref_bl, on=["faixa_velocidade_bl", "CLASSIFICACAO"])
    df["sobrepreco_bl"] = np.where(tem_valor_bl, df["VALOR_BL_M0"] / df["ref_bl"] - 1, np.nan)

    tem_valor_tv = df["VALOR_TV_M0"] > 0
    ref_tv = df[tem_valor_tv].groupby("tier_tv")["VALOR_TV_M0"].median().rename("ref_tv")
    df = df.join(ref_tv, on="tier_tv")
    df["sobrepreco_tv"] = np.where(tem_valor_tv, df["VALOR_TV_M0"] / df["ref_tv"] - 1, np.nan)
    df = df.drop(columns=["ref_bl", "ref_tv"])

    # Tempo
    dt = df["DATA_FUNIL_SOLAR"]
    df["mes"] = dt.dt.to_period("M").astype(str)
    df["dia_semana"] = dt.dt.dayofweek
    df["hora"] = dt.dt.hour

    # Reincidência: quantas vezes o cliente já tinha entrado no funil antes
    # deste lead (só olhando o passado, para não vazar informação)
    df["n_leads_anteriores"] = df.groupby("id_cliente").cumcount()
    anterior = df.groupby("id_cliente")["DATA_FUNIL_SOLAR"].shift()
    df["dias_desde_lead_anterior"] = (dt - anterior).dt.days

    # Porte da praça (volume de leads da cidade na base): proxy de cidade grande
    df["leads_na_cidade"] = df.groupby("COD_IBGE")["id_lead"].transform("size")

    return df


# ---------------------------------------------------------------------------
# Desfecho (o que aconteceu até M30) - NUNCA usar como variável explicativa
# ---------------------------------------------------------------------------
def criar_desfechos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["status"] = df["NM_SUB_INDICADOR_NEGOCIO"]
    df["retido"] = (df["status"] == "RETIDO").astype(int)
    df["movimentou"] = df[[f"FLAG_MOV_{p}" for p in PRODUTOS]].max(axis=1)
    df["cancelou_algum"] = df[[f"FLAG_CANC_{p}" for p in PRODUTOS]].max(axis=1)
    df["qtd_produtos_movimentados"] = df[[f"FLAG_MOV_{p}" for p in PRODUTOS]].sum(axis=1)

    # Alvo pedido no case: cliente retido que alterou ao menos um produto
    df["alvo_retido_mov"] = ((df["retido"] == 1) & (df["movimentou"] == 1)).astype(int)

    df["receita_m30"] = df[[f"VALOR_{p}_M30" for p in PRODUTOS]].sum(axis=1)
    df["delta_receita"] = df["receita_m30"] - df["receita_m0"]
    df["receita_preservada_pct"] = np.where(
        df["receita_m0"] > 0, df["receita_m30"] / df["receita_m0"], np.nan
    )

    for p in PRODUTOS:
        df[f"delta_{p.lower()}"] = df[f"VALOR_{p}_M30"] - df[f"VALOR_{p}_M0"]

    # Mesmo produto, preço diferente: o "desconto sem troca"
    mesmo_produto = np.ones(len(df), dtype=bool)
    for p in PRODUTOS:
        mesmo_produto &= df[COL_ID[p].format(m="M0")].values == df[COL_ID[p].format(m="M30")].values

    susp_m30 = df["DSC_PRODUTO_BL_M30"].str.contains("SUSPENS") & (df["bl_suspensa_m0"] == 0)
    mov = df["movimentou"] == 1
    canc = df["cancelou_algum"] == 1
    d = df["delta_receita"]

    cond = [
        susp_m30,
        mov & canc,
        mov & (d < -1),
        mov & (d > 1),
        mov,
        canc,
        mesmo_produto & (d < -1),
        mesmo_produto & (d > 1),
    ]
    rot = [
        "Suspensao temporaria",
        "Troca + cancelamento parcial",
        "Troca com reducao de valor",
        "Troca com aumento de valor",
        "Troca sem impacto no valor",
        "Cancelamento parcial",
        "Desconto sem troca",
        "Reajuste sem troca",
    ]
    df["tipo_desfecho"] = np.select(cond, rot, "Sem alteracao")

    # Na banda larga dá para saber se a troca foi para mais ou menos velocidade
    vel_m30 = _velocidade_mbps(df["DSC_PRODUTO_BL_M30"])
    df["velocidade_bl_m30"] = vel_m30
    troca_bl = df["FLAG_MOV_BL"] == 1
    df["direcao_velocidade_bl"] = np.select(
        [troca_bl & (vel_m30 > df["velocidade_bl_m0"]),
         troca_bl & (vel_m30 < df["velocidade_bl_m0"]),
         troca_bl & (vel_m30 == df["velocidade_bl_m0"])],
        ["upgrade", "downgrade", "mesma velocidade"],
        "",
    )
    df["direcao_velocidade_bl"] = np.where(troca_bl & (df["direcao_velocidade_bl"] == ""),
                                           "indefinido", df["direcao_velocidade_bl"])
    return df


def preparar_base(usar_cache: bool = True) -> pd.DataFrame:
    """Pipeline completo: leitura -> limpeza -> variáveis M0 -> desfechos."""
    df = carregar_base(usar_cache)
    df = limpar(df)
    df = criar_variaveis_m0(df)
    df = criar_desfechos(df)
    return df


# Variáveis que podem entrar no modelo: todas conhecidas no momento M0
VARS_NUMERICAS = [
    "tem_bl", "tem_tv", "tem_fone", "tem_mv", "qtd_produtos", "flag_movel",
    "VALOR_BL_M0", "VALOR_TV_M0", "VALOR_MV_M0", "receita_m0", "peso_tv_no_ticket",
    "sobrepreco_bl", "sobrepreco_tv",
    "velocidade_bl_m0", "bl_suspensa_m0", "bl_gplay", "bl_pon", "tv_streaming",
    "plano_de_retencao_m0", "plano_old_m0", "plano_migracao_m0",
    "FLG_BROWNFIELD", "n_leads_anteriores", "dias_desde_lead_anterior",
    "leads_na_cidade", "dia_semana", "hora",
]
VARS_CATEGORICAS = [
    "combo_m0", "tier_tv", "tipo_movel", "faixa_velocidade_bl", "DSC_CLI_MPLAY_M0",
    "NM_GRUPO_REGIONAL", "SGL_UF", "CLASSIFICACAO", "TP_REDE_HP", "FLAG_ORIGEM",
]
