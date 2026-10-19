"""
Dashboard do case de retenção.

Rodar a partir da raiz do projeto:
    streamlit run app/streamlit_app.py

O app usa só tabelas agregadas e os modelos exportados no final do notebook
(pasta outputs/). Nenhuma linha de cliente é carregada.
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore", message=".*TreeExplainer shap values output has changed.*")

RAIZ = Path(__file__).resolve().parents[1]
DIR_APP = RAIZ / "outputs" / "app"
DIR_MOD = RAIZ / "outputs" / "modelos"

AZUL, LARANJA, VERDE_AGUA, AMARELO, MAGENTA, VERDE, VIOLETA, VERMELHO = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
CINZA, CINZA_CLARO, TINTA_2 = "#898781", "#c3c2b7", "#52514e"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

st.set_page_config(page_title="Retenção com movimentação", layout="wide", page_icon="📶")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px;}
div[data-testid="stMetric"] {background: rgba(127,127,127,0.06); border: 1px solid rgba(127,127,127,0.18);
  border-radius: 10px; padding: 12px 16px;}
div[data-testid="stMetricLabel"] p {font-size: 0.82rem; opacity: 0.8;}
div[data-testid="stMetricValue"] {font-size: 1.65rem;}
.nota {font-size: 0.86rem; opacity: 0.75;}
h3 {margin-top: 0.6rem;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
@st.cache_data
def carregar():
    ler = lambda n: pd.read_parquet(DIR_APP / f"{n}.parquet")
    nomes = ["cubo", "cidades", "ref_bl", "ref_tv", "perfis", "tipologia", "efeito_solar", "balanceamento",
             "comparacao_modelos", "decis", "importancia_shap", "matriz_velocidade", "matriz_tv",
             "curva_sobrepreco", "faixas_risco", "projecao_solar"]
    dados = {n: ler(n) for n in nomes}
    dados["meta"] = json.loads((DIR_APP / "meta.json").read_text(encoding="utf-8"))
    return dados


@st.cache_resource
def carregar_modelos():
    return joblib.load(DIR_MOD / "lgbm_retido_mov.pkl"), joblib.load(DIR_MOD / "lgbm_triagem.pkl")


if not (DIR_APP / "meta.json").exists():
    st.error("Arquivos do app não encontrados. Execute o notebook até a seção de exportação.")
    st.stop()

D = carregar()
META = D["meta"]
CUBO = D["cubo"]


def layout(fig, altura=380, legenda=True):
    tem_titulo = bool(fig.layout.title.text)
    fig.update_layout(height=altura, margin=dict(l=10, r=10, t=78 if (legenda and tem_titulo) else 44, b=10),
                      font=dict(family="system-ui, Segoe UI, sans-serif", size=12),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      showlegend=legenda, legend=dict(orientation="h", y=1.01, yanchor="bottom", x=0),
                      hoverlabel=dict(font_size=12), separators=",.")
    if tem_titulo:
        fig.update_layout(title=dict(y=0.98, yanchor="top", x=0, xanchor="left"))
    fig.update_xaxes(showgrid=False, linecolor=CINZA_CLARO)
    fig.update_yaxes(gridcolor="rgba(137,135,129,0.25)", zeroline=False)
    return fig


def reais(v, casas=0):
    return ("R$ " + f"{v:,.{casas}f}").replace(",", "X").replace(".", ",").replace("X", ".")


def num(v, casas=0):
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(v, casas=1):
    return f"{v:.{casas}%}".replace(".", ",")


# ---------------------------------------------------------------------------
# Barra lateral: filtros (valem para as abas descritivas)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Filtros")
    meses = sorted(CUBO["mes"].unique())
    rotulo_mes = {m: f"{m[5:]}/{m[2:4]}" for m in meses}
    ini, fim = st.select_slider("Período", options=meses, value=(meses[0], meses[-1]),
                                format_func=rotulo_mes.get)
    origens = st.multiselect("Origem do atendimento", sorted(CUBO["FLAG_ORIGEM"].unique()),
                             default=sorted(CUBO["FLAG_ORIGEM"].unique()))
    regionais = st.multiselect("Regional", sorted(CUBO["NM_GRUPO_REGIONAL"].unique()),
                               placeholder="Todas")
    combos = st.multiselect("Combo em M0", sorted(CUBO["combo_m0"].unique()), placeholder="Todos")
    st.markdown("---")
    st.markdown('<p class="nota">Os filtros valem para as abas Visão geral, Perfis e Movimentações. '
                'Valores de receita são mensalidades (receita recorrente mensal) da amostra.</p>',
                unsafe_allow_html=True)

f = CUBO[(CUBO["mes"] >= ini) & (CUBO["mes"] <= fim) & (CUBO["FLAG_ORIGEM"].isin(origens))]
if regionais:
    f = f[f["NM_GRUPO_REGIONAL"].isin(regionais)]
if combos:
    f = f[f["combo_m0"].isin(combos)]
if f.empty:
    st.warning("Nenhum lead com esses filtros.")
    st.stop()

fechados = f[f["status"] != "LEAD EM ABERTO"]
ret = f[f["status"] == "RETIDO"]

st.title("Retenção com movimentação")
st.markdown('<p class="nota">Quem aceita trocar de produto para ficar, quanto isso custa e onde agir. '
            'Amostra de leads do funil de retenção, jun–dez/2025.</p>', unsafe_allow_html=True)

aba = st.tabs(["Visão geral", "Perfis e drivers", "Movimentações", "SOLAR x Legado",
               "Simulador de atendimento", "Sobre o modelo"])

# ---------------------------------------------------------------------------
# 1. Visão geral
# ---------------------------------------------------------------------------
with aba[0]:
    leads = f["leads"].sum()
    taxa_ret = fechados["retidos"].sum() / max(fechados["leads"].sum(), 1)
    taxa_mov = ret["retido_mov"].sum() / max(ret["leads"].sum(), 1)
    rec_ent, rec_sai = fechados["receita_m0"].sum(), fechados["receita_m30"].sum()

    c = st.columns(5)
    c[0].metric("Leads no funil", num(leads))
    c[1].metric("Taxa de retenção", pct(taxa_ret))
    c[2].metric("Mov. entre retidos", pct(taxa_mov))
    c[3].metric("Receita preservada", pct(rec_sai / rec_ent))
    c[4].metric("Receita perdida / mês", reais((rec_ent - rec_sai) / 1e6, 1) + " mi")

    col1, col2 = st.columns([1, 1.25])
    with col1:
        etapas = ["Entraram no funil", "Finalizados", "Retidos", "Retidos com movimentação"]
        vals = [leads, fechados["leads"].sum(), ret["leads"].sum(), ret["retido_mov"].sum()]
        fig = go.Figure(go.Funnel(y=etapas, x=vals, texttemplate="%{value:,.0f}<br>%{percentInitial:.0%}",
                                  marker=dict(color=[CINZA_CLARO, CINZA_CLARO, AZUL, LARANJA]),
                                  connector=dict(line=dict(color="rgba(0,0,0,0)"))))
        fig.update_layout(title="Funil")
        st.plotly_chart(layout(fig, 360, False), width="stretch")

    with col2:
        m = (f.groupby("mes").apply(lambda g: pd.Series({
            "Retenção": g.loc[g.status != "LEAD EM ABERTO", "retidos"].sum() / max(g.loc[g.status != "LEAD EM ABERTO", "leads"].sum(), 1),
            "Movimentação entre retidos": g.loc[g.status == "RETIDO", "retido_mov"].sum() / max(g.loc[g.status == "RETIDO", "leads"].sum(), 1),
            "Receita preservada": g.loc[g.status != "LEAD EM ABERTO", "receita_m30"].sum() / max(g.loc[g.status != "LEAD EM ABERTO", "receita_m0"].sum(), 1),
        }), include_groups=False))
        fig = go.Figure()
        for col, cor in zip(m.columns, [AZUL, LARANJA, VERDE_AGUA]):
            fig.add_trace(go.Scatter(x=[rotulo_mes[i] for i in m.index], y=m[col], name=col, mode="lines+markers",
                                     line=dict(color=cor, width=2), marker=dict(size=7),
                                     hovertemplate="%{x}: %{y:.1%}<extra>" + col + "</extra>"))
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        fig.update_layout(title="Mês a mês", hovermode="x unified")
        st.plotly_chart(layout(fig, 360), width="stretch")

    st.markdown("### Para onde vai a receita do funil")
    comp = np.select(
        [fechados["status"] == "NAO RETIDO",
         fechados["receita_zerada_m30"],
         fechados["tipo_desfecho"].isin(["Cancelamento parcial", "Troca + cancelamento parcial", "Suspensao temporaria"]),
         fechados["tipo_desfecho"].str.startswith("Troca"),
         fechados["tipo_desfecho"] == "Desconto sem troca",
         fechados["tipo_desfecho"] == "Reajuste sem troca"],
        ["Não retidos", "Retidos c/ conta zerada", "Cancel. parcial / suspensão", "Troca de produto",
         "Desconto sem troca", "Reajuste sem troca"],
        "Sem alteração")
    delta = (fechados["receita_m30"] - fechados["receita_m0"]).groupby(comp).sum()
    ordem = ["Não retidos", "Retidos c/ conta zerada", "Cancel. parcial / suspensão", "Troca de produto",
             "Desconto sem troca", "Reajuste sem troca"]
    delta = delta.reindex(ordem).fillna(0)
    fig = go.Figure(go.Waterfall(
        x=["Receita M0"] + ordem + ["Receita M30"],
        measure=["absolute"] + ["relative"] * len(ordem) + ["total"],
        y=[rec_ent] + list(delta.values) + [0],
        text=[reais(rec_ent / 1e6, 1) + " mi"] + [("–" if v < 0 else "+") + pct(abs(v) / rec_ent) for v in delta.values] + [pct(rec_sai / rec_ent) + " da entrada"],
        textposition="outside",
        decreasing=dict(marker=dict(color=VERMELHO)), increasing=dict(marker=dict(color=VERDE)),
        totals=dict(marker=dict(color=AZUL)), connector=dict(line=dict(color=CINZA_CLARO, width=1)),
        hovertemplate="%{x}<br>R$ %{y:,.0f}<extra></extra>"))
    fig.update_yaxes(tickprefix="R$ ", tickformat=",.0f", range=[0, rec_ent * 1.12])
    st.plotly_chart(layout(fig, 420, False), width="stretch")
    st.markdown('<p class="nota">Os retidos levam mais receita que os não retidos: clientes marcados como RETIDO que zeram '
                'a conta em 30 dias, cancelamentos parciais e trocas. Reter não é sinônimo de preservar receita.</p>',
                unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2. Perfis e drivers
# ---------------------------------------------------------------------------
with aba[1]:
    dims = {"combo_m0": "Combo em M0", "tier_tv": "Pacote de TV", "faixa_velocidade_bl": "Velocidade da BL",
            "NM_GRUPO_REGIONAL": "Regional", "SGL_UF": "UF", "CLASSIFICACAO": "Classificação da praça",
            "TP_REDE_HP": "Rede", "FLAG_ORIGEM": "Origem"}
    col1, col2 = st.columns([1, 1])
    with col1:
        dim = st.selectbox("Abrir taxa de movimentação (entre retidos) por", list(dims), format_func=dims.get)
        t = ret.groupby(dim).agg(leads=("leads", "sum"), mov=("retido_mov", "sum"))
        t = t[t["leads"] >= 300]
        t["taxa"] = t["mov"] / t["leads"]
        t = t.sort_values("taxa")
        fig = go.Figure(go.Bar(x=t["taxa"], y=t.index.astype(str), orientation="h", marker_color=AZUL,
                               text=[pct(v) for v in t["taxa"]], textposition="outside",
                               customdata=t["leads"], hovertemplate="%{y}<br>Taxa: %{x:.1%}<br>Leads: %{customdata:,}<extra></extra>"))
        fig.add_vline(x=taxa_mov, line_dash="dash", line_color=TINTA_2, annotation_text="média",
                      annotation_position="top")
        fig.update_xaxes(tickformat=".0%", range=[0, max(t["taxa"].max() * 1.25, 0.5)])
        st.plotly_chart(layout(fig, max(300, 30 * len(t) + 80), False), width="stretch")

    with col2:
        cv = D["curva_sobrepreco"]
        fig = go.Figure(go.Scatter(x=cv["sobrepreco_medio"], y=cv["taxa"], mode="lines+markers",
                                   line=dict(color=AZUL, width=2), marker=dict(size=8),
                                   customdata=cv["leads"],
                                   hovertemplate="Sobrepreço %{x:+.0%}<br>Taxa %{y:.1%}<br>%{customdata:,} leads<extra></extra>"))
        fig.add_vline(x=0, line_color=CINZA)
        fig.update_xaxes(tickformat="+.0%", title="Valor da BL vs. preço típico (mesma velocidade e praça)")
        fig.update_yaxes(tickformat=".0%", title="Taxa de movimentação")
        fig.update_layout(title="Quem paga acima do preço típico aceita mais a troca")
        st.plotly_chart(layout(fig, 380, False), width="stretch")

        imp = D["importancia_shap"].head(12).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp["shap"], y=imp.index, orientation="h", marker_color=AZUL,
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_layout(title="Importância das variáveis no modelo (|SHAP| médio)")
        st.plotly_chart(layout(fig, 400, False), width="stretch")

    st.markdown("### Perfis acionáveis")
    st.markdown('<p class="nota">Segmentos de uma árvore de decisão rasa treinada em jun–out; taxas conferidas em nov–dez. '
                'Tamanho da bolha = volume de leads retidos.</p>', unsafe_allow_html=True)
    p = D["perfis"].copy()
    grupos = {**dict.fromkeys(["P01", "P03", "P04", "P08"], "Convergente no preço de tabela ou acima"),
              **dict.fromkeys(["P02", "P06", "P07", "P10"], "Residencial sem móvel, BL cara"),
              **dict.fromkeys(["P05", "P09", "P11"], "Portfólio grande com preço já bom"),
              **dict.fromkeys(["P12", "P13", "P14"], "Single-play barato")}
    p["grupo"] = p["perfil"].map(grupos)
    cores_grupo = {g: c for g, c in zip(dict.fromkeys(grupos.values()), [AZUL, LARANJA, VERDE_AGUA, VIOLETA])}
    fig = go.Figure()
    for g, cor in cores_grupo.items():
        q = p[p["grupo"] == g]
        fig.add_trace(go.Scatter(
            x=q["taxa_treino"], y=q["ticket_medio"], mode="markers+text", text=q["perfil"], name=g,
            textposition="middle center", textfont=dict(size=10, color="white"),
            marker=dict(size=np.sqrt(q["leads"]) / 2.4, color=cor, opacity=0.85, line=dict(width=2, color="white")),
            customdata=np.stack([q["regra"], q["leads"], q["taxa_teste"]], axis=1),
            hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>Leads: %{customdata[1]:,}<br>"
                          "Taxa treino: %{x:.1%} | teste: %{customdata[2]:.1%}<br>Ticket: R$ %{y:.0f}<extra></extra>"))
    fig.add_vline(x=META["taxa_mov_treino"], line_dash="dash", line_color=TINTA_2, annotation_text="média")
    fig.update_xaxes(tickformat=".0%", title="Taxa de movimentação")
    fig.update_yaxes(tickprefix="R$ ", title="Ticket médio M0")
    st.plotly_chart(layout(fig, 480), width="stretch")

    tab = p[["perfil", "grupo", "regra", "leads", "taxa_teste", "lift", "ticket_medio", "delta_receita_movimentadores"]].copy()
    tab.columns = ["Perfil", "Grupo", "Regra", "Leads", "Taxa (teste)", "Lift", "Ticket M0", "Δ receita de quem movimenta"]
    st.dataframe(tab, hide_index=True, width="stretch", height=530,
                 column_config={"Taxa (teste)": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
                                "Lift": st.column_config.NumberColumn(format="%.2fx"),
                                "Ticket M0": st.column_config.NumberColumn(format="R$ %.0f"),
                                "Δ receita de quem movimenta": st.column_config.NumberColumn(format="R$ %.0f"),
                                "Regra": st.column_config.TextColumn(width="large")})

# ---------------------------------------------------------------------------
# 3. Movimentações
# ---------------------------------------------------------------------------
with aba[2]:
    tp = ret.groupby("tipo_desfecho").agg(leads=("leads", "sum"), r0=("receita_m0", "sum"), r30=("receita_m30", "sum"))
    tp["delta_medio"] = (tp["r30"] - tp["r0"]) / tp["leads"]
    tp["pct"] = tp["leads"] / tp["leads"].sum()
    tp["preservada"] = tp["r30"] / tp["r0"]
    tp = tp.sort_values("delta_medio")
    nomes_tipo = {"Suspensao temporaria": "Suspensão temporária", "Troca com reducao de valor": "Troca com redução de valor",
                  "Sem alteracao": "Sem alteração"}
    col1, col2 = st.columns([1.2, 1])
    with col1:
        cores = [VERMELHO if v < -1 else (VERDE if v > 1 else CINZA_CLARO) for v in tp["delta_medio"]]
        fig = go.Figure(go.Bar(x=tp["delta_medio"], y=[nomes_tipo.get(i, i) for i in tp.index], orientation="h",
                               marker_color=cores, text=[f"{reais(v)} · {pct(s, 0)}" for v, s in zip(tp["delta_medio"], tp["pct"])],
                               textposition="outside",
                               hovertemplate="%{y}<br>Variação média: R$ %{x:.2f}<extra></extra>"))
        fig.update_layout(title="Variação média da mensalidade por tipo de desfecho (retidos)")
        fig.update_xaxes(tickprefix="R$ ", range=[tp["delta_medio"].min() * 1.6, max(tp["delta_medio"].max() * 3, 60)])
        st.plotly_chart(layout(fig, 420, False), width="stretch")
    with col2:
        tabela = tp[["leads", "pct", "preservada", "delta_medio"]].sort_values("leads", ascending=False).reset_index()
        tabela["tipo_desfecho"] = tabela["tipo_desfecho"].map(lambda i: nomes_tipo.get(i, i))
        tabela.columns = ["Tipo", "Leads", "% dos retidos", "Receita preservada", "Δ médio (R$)"]
        st.dataframe(tabela, hide_index=True, width="stretch", height=380,
                     column_config={"% dos retidos": st.column_config.NumberColumn(format="%.3f"),
                                    "Receita preservada": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1.2),
                                    "Δ médio (R$)": st.column_config.NumberColumn(format="%.2f")})
        st.markdown('<p class="nota">Régua sugerida de ofertas (da que mais preserva para a que menos preserva receita): '
                    'troca com aumento → troca neutra → desconto sem troca → troca com redução → cancelamento parcial / suspensão.</p>',
                    unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    for col, nome, titulo in [(col1, "matriz_velocidade", "Troca de BL: velocidade M0 → M30 (% da linha)"),
                              (col2, "matriz_tv", "Troca de TV: pacote M0 → M30 (% da linha)")]:
        mtx = D[nome].fillna(0)
        fig = go.Figure(go.Heatmap(z=mtx.values, x=list(mtx.columns.astype(str)), y=list(mtx.index.astype(str)),
                                   colorscale=SEQ, showscale=False, text=[[pct(v, 0) for v in r] for r in mtx.values],
                                   texttemplate="%{text}", hovertemplate="De %{y} para %{x}: %{z:.1%}<extra></extra>",
                                   xgap=2, ygap=2))
        fig.update_yaxes(autorange="reversed", title="M0")
        fig.update_xaxes(title="M30")
        fig.update_layout(title=titulo)
        col.plotly_chart(layout(fig, 400, False), width="stretch")
    st.markdown('<p class="nota">Na BL, a troca é majoritariamente para a mesma velocidade ou para cima, com preço menor: '
                'o catálogo atual entrega mais velocidade pelo mesmo dinheiro e o retentor usa isso como argumento.</p>',
                unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 4. SOLAR x Legado
# ---------------------------------------------------------------------------
with aba[3]:
    ef = D["efeito_solar"]
    pj = D["projecao_solar"]["estimativa"]
    st.markdown("A comparação direta entre SOLAR e Legado mistura o efeito da ferramenta com o perfil de cliente que cada "
                "um atende. Aqui os leads foram **reponderados por escore de propensão** para comparar clientes equivalentes.")
    c = st.columns(4)
    c[0].metric("Efeito na retenção (ajustado)", num(ef.loc['Taxa de retenção', 'efeito ajustado (ATT)'] * 100, 1).join(["+", " p.p."]),
                delta="bruto: +" + num(ef.loc['Taxa de retenção', 'diferença bruta'] * 100, 1) + " p.p.", delta_color="off")
    c[1].metric("Efeito na receita por lead", reais(ef.loc["Receita M30 por lead (R$)", "efeito ajustado (ATT)"], 2),
                delta="bruto: " + reais(ef.loc['Receita M30 por lead (R$)', 'diferença bruta'], 2), delta_color="off")
    c[2].metric("Legado → SOLAR: retidos/mês", "+" + num(pj["Clientes a mais retidos por mês"]),
                delta="+" + num(pj['Ganho de retenção (p.p.)'], 1) + " p.p.", delta_color="off")
    c[3].metric("Legado → SOLAR: receita/mês", reais(pj["Variação de receita mensal (R$)"]),
                delta="desconto mais profundo", delta_color="off")

    col1, col2 = st.columns(2)
    with col1:
        e = ef.loc[["Taxa de retenção", "Retido com movimentação"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Diferença bruta", x=e.index, y=e["diferença bruta"] * 100, marker_color=CINZA_CLARO,
                             hovertemplate="%{x}: %{y:+.1f} p.p.<extra>bruta</extra>"))
        fig.add_trace(go.Bar(name="Efeito ajustado (IC 95%)", x=e.index, y=e["efeito ajustado (ATT)"] * 100, marker_color=AZUL,
                             error_y=dict(type="data", symmetric=False, color=TINTA_2,
                                          array=(e["IC 95% sup"] - e["efeito ajustado (ATT)"]) * 100,
                                          arrayminus=(e["efeito ajustado (ATT)"] - e["IC 95% inf"]) * 100),
                             hovertemplate="%{x}: %{y:+.1f} p.p.<extra>ajustado</extra>"))
        fig.update_yaxes(ticksuffix=" p.p.")
        fig.update_layout(title="SOLAR vs. Legado: bruto x ajustado", barmode="group")
        st.plotly_chart(layout(fig, 400), width="stretch")
    with col2:
        b = D["balanceamento"].sort_values("antes").tail(20)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=b["antes"], y=b.index, mode="markers", name="Antes", marker=dict(color=LARANJA, size=9)))
        fig.add_trace(go.Scatter(x=b["depois (ATT)"], y=b.index, mode="markers", name="Depois", marker=dict(color=AZUL, size=9)))
        fig.add_vline(x=0.1, line_dash="dash", line_color=TINTA_2)
        fig.update_layout(title="Balanceamento das variáveis (|diferença padronizada|)")
        st.plotly_chart(layout(fig, 400), width="stretch")
    st.markdown('<p class="nota">Para o perfil já atendido pelo SOLAR, a ferramenta retém mais e preserva mais receita. '
                'Para o perfil Legado, retém +5 p.p. mas com receita por lead neutra a levemente menor: expandir com piso de '
                'preço. Estudo observacional (corrige diferenças observadas na base); confirmar com teste A/B.</p>',
                unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 5. Simulador
# ---------------------------------------------------------------------------
with aba[4]:
    gbm, triagem = carregar_modelos()
    cid = D["cidades"].sort_values("leads_na_cidade", ascending=False)
    cat = META["categorias"]

    st.markdown("Monte um cliente como ele chega ao funil (informações de **M0**) e veja o que os modelos esperam e "
                "qual abordagem faz mais sentido.")
    col_in, col_out = st.columns([1, 1.35], gap="large")
    with col_in:
        c1, c2 = st.columns(2)
        uf = c1.selectbox("UF", sorted(cid["SGL_UF"].unique()), index=sorted(cid["SGL_UF"].unique()).index("SP"))
        cidades_uf = cid[cid["SGL_UF"] == uf]
        cidade = c2.selectbox("Cidade", cidades_uf["DSC_CIDADE"])
        info = cidades_uf[cidades_uf["DSC_CIDADE"] == cidade].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        tem_bl = c1.checkbox("BL", True)
        tem_tv = c2.checkbox("TV", True)
        tem_fone = c3.checkbox("Fone", False)
        tem_mv = c4.checkbox("Móvel", False)

        c1, c2 = st.columns(2)
        vel = c1.select_slider("Velocidade da BL (Mega)", [100, 250, 300, 350, 400, 500, 600, 750, 1000], 500, disabled=not tem_bl)
        valor_bl = c2.number_input("Valor da BL (R$)", 0.0, 500.0, 110.0, 5.0, disabled=not tem_bl)
        c1, c2 = st.columns(2)
        tier = c1.selectbox("Pacote de TV", ["ENTRADA", "MIX", "TOP", "TOP 4K"], 2, disabled=not tem_tv)
        valor_tv = c2.number_input("Valor da TV (R$)", 0.0, 600.0, 140.0, 5.0, disabled=not tem_tv)
        streaming = st.checkbox("TV já é streaming / CTV+", False, disabled=not tem_tv)
        c1, c2 = st.columns(2)
        tipo_mv = c1.selectbox("Plano móvel", ["POS", "CONTROLE"], disabled=not tem_mv)
        valor_mv = c2.number_input("Valor do móvel (R$)", 0.0, 400.0, 80.0, 5.0, disabled=not tem_mv)

        c1, c2 = st.columns(2)
        origem = c1.radio("Atendimento", ["SOLAR", "LEGADO"], horizontal=True)
        flag_movel = c2.radio("Cliente tem móvel na operadora?", ["Não", "Sim"], horizontal=True, index=1 if tem_mv else 0)
        c1, c2 = st.columns(2)
        plano_ret = c1.checkbox("Já está em plano de retenção (RET)")
        reinc = c2.number_input("Leads anteriores no funil", 0, 8, 0)
        hora = st.slider("Hora do contato", 0, 23, 14)

    # Monta a linha no mesmo formato do treino
    def faixa(v):
        return "ate 250M" if v < 300 else "300-350M" if v < 400 else "400-500M" if v < 550 else "600-750M" if v < 800 else "1G ou mais"

    fx = faixa(vel) if tem_bl else "sem BL / suspensa"
    rb = D["ref_bl"]
    ref_bl = rb[(rb["faixa_velocidade_bl"] == fx) & (rb["CLASSIFICACAO"] == info["classificacao"])]["ref"]
    ref_bl = float(ref_bl.iloc[0]) if len(ref_bl) else np.nan
    rt = D["ref_tv"].set_index("tier_tv")["ref"]
    vbl, vtv, vmv = (valor_bl if tem_bl else 0.0), (valor_tv if tem_tv else 0.0), (valor_mv if tem_mv else 0.0)
    receita = vbl + vtv + vmv
    combo = "+".join([p for p, t in zip(["BL", "TV", "FONE", "MV"], [tem_bl, tem_tv, tem_fone, tem_mv]) if t]) or "NENHUM"

    linha = {
        "tem_bl": int(tem_bl), "tem_tv": int(tem_tv), "tem_fone": int(tem_fone), "tem_mv": int(tem_mv),
        "qtd_produtos": int(tem_bl + tem_tv + tem_fone + tem_mv), "flag_movel": int(flag_movel == "Sim"),
        "VALOR_BL_M0": vbl, "VALOR_TV_M0": vtv, "VALOR_MV_M0": vmv, "receita_m0": receita,
        "peso_tv_no_ticket": vtv / receita if receita > 0 else 0,
        "sobrepreco_bl": vbl / ref_bl - 1 if tem_bl and vbl > 0 and ref_bl == ref_bl else np.nan,
        "sobrepreco_tv": vtv / rt.get(tier, np.nan) - 1 if tem_tv and vtv > 0 else np.nan,
        "velocidade_bl_m0": float(vel) if tem_bl else np.nan, "bl_suspensa_m0": 0, "bl_gplay": 1,
        "bl_pon": int(info["rede"] == "GPON"), "tv_streaming": int(streaming and tem_tv),
        "plano_de_retencao_m0": int(plano_ret), "plano_old_m0": 0, "plano_migracao_m0": 0,
        "FLG_BROWNFIELD": 0, "n_leads_anteriores": reinc, "dias_desde_lead_anterior": 45.0 if reinc else np.nan,
        "leads_na_cidade": info["leads_na_cidade"], "dia_semana": 2, "hora": hora,
        "combo_m0": combo, "tier_tv": tier if tem_tv else "sem TV", "tipo_movel": tipo_mv if tem_mv else "sem plano",
        "faixa_velocidade_bl": fx, "DSC_CLI_MPLAY_M0": "NAO ELEGIVEL", "NM_GRUPO_REGIONAL": info["regional"],
        "SGL_UF": uf, "CLASSIFICACAO": info["classificacao"], "TP_REDE_HP": info["rede"], "FLAG_ORIGEM": origem,
    }
    Xs = pd.DataFrame([linha])[META["vars_numericas"] + META["vars_categoricas"]]
    for c in META["vars_categoricas"]:
        Xs[c] = pd.Categorical(Xs[c], categories=cat[c])

    p_mov = float(gbm.predict_proba(Xs)[0, 1])
    p3 = triagem.predict_proba(Xs)[0]

    with col_out:
        c = st.columns(3)
        c[0].metric("Chance de não reter", pct(p3[0], 0))
        c[1].metric("Retido sem troca", pct(p3[1], 0))
        c[2].metric("Retido com troca", pct(p3[2], 0))
        fig = go.Figure()
        for nome, v, cor in zip(META["classes_triagem"], p3, [VERMELHO, CINZA_CLARO, LARANJA]):
            fig.add_trace(go.Bar(x=[v], y=[""], orientation="h", name=nome, marker_color=cor,
                                 text=pct(v, 0), textposition="inside", hovertemplate=nome + ": %{x:.1%}<extra></extra>"))
        fig.update_layout(barmode="stack", xaxis=dict(tickformat=".0%", range=[0, 1], showgrid=False),
                          legend=dict(traceorder="normal"))
        st.plotly_chart(layout(fig, 150), width="stretch")

        st.metric("Se ficar, chance de aceitar troca de produto (modelo do case)", pct(p_mov, 0),
                  delta=f"{(p_mov / META['taxa_mov_treino'] - 1) * 100:+.0f}% vs. média dos retidos".replace(".", ","),
                  delta_color="off")

        # Recomendação: regras simples em cima dos achados do estudo
        recs = []
        if linha["sobrepreco_bl"] == linha["sobrepreco_bl"] and linha["sobrepreco_bl"] > 0.03:
            recs.append(f"**Oferta de troca de BL.** O cliente paga {pct(linha['sobrepreco_bl'], 0)} acima do preço típico para "
                        f"{fx} na praça ({reais(ref_bl, 2)}). Há espaço para migrar para um plano atual de mesma ou maior "
                        f"velocidade perto do preço de referência – é o padrão de troca mais comum entre os retidos.")
        elif tem_bl and linha["sobrepreco_bl"] == linha["sobrepreco_bl"]:
            recs.append("**BL já está com preço bom.** Clientes abaixo do preço típico raramente aceitam troca; "
                        "priorizar argumentos de valor (velocidade, serviços inclusos) em vez de preço.")
        if tem_tv and linha["peso_tv_no_ticket"] > 0.45 and not streaming:
            recs.append(f"**TV pesa {pct(linha['peso_tv_no_ticket'], 0)} da conta.** Ofertar migração do pacote linear para "
                        "streaming / CTV+ (em geral de nível Top) com mensalidade menor: 87% das trocas de TV no estudo seguem esse caminho.")
        if plano_ret:
            recs.append("**Já está em plano de retenção.** Menos espaço para nova redução de preço; considerar benefício "
                        "não monetário ou upgrade de velocidade.")
        if p3[0] > META["limiar_risco_alto"]:
            recs.append("**Risco alto de perda.** Faixa em que se concentra a maior parte dos não retidos: vale acionar a "
                        "oferta mais forte disponível e evitar terminar em cancelamento parcial.")
        if origem == "LEGADO":
            recs.append("**Atendimento via Legado.** No estudo, clientes equivalentes atendidos pelo SOLAR retêm mais; "
                        "se possível, usar a recomendação do SOLAR.")
        if not recs:
            recs.append("**Perfil de baixa propensão à troca.** Retenção tende a acontecer sem mexer no portfólio; "
                        "evitar conceder desconto que não foi pedido.")
        st.markdown("#### Abordagem sugerida")
        for r in recs:
            st.markdown("- " + r)

        with st.expander("Por que o modelo chegou nessa probabilidade? (contribuições SHAP)"):
            import shap
            expl = shap.TreeExplainer(gbm)
            sv = expl.shap_values(Xs)
            sv = sv[1] if isinstance(sv, list) else sv
            contrib = pd.Series(sv[0], index=Xs.columns).sort_values(key=np.abs, ascending=False).head(10).iloc[::-1]
            fig = go.Figure(go.Bar(x=contrib.values, y=contrib.index, orientation="h",
                                   marker_color=[AZUL if v > 0 else LARANJA for v in contrib.values],
                                   hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
            fig.add_vline(x=0, line_color=TINTA_2)
            fig.update_layout(title="Azul empurra para movimentar; laranja, para não movimentar")
            st.plotly_chart(layout(fig, 380, False), width="stretch")

# ---------------------------------------------------------------------------
# 6. Sobre o modelo
# ---------------------------------------------------------------------------
with aba[5]:
    st.markdown("""
**Desenho**: população = leads RETIDOS; alvo = troca de ao menos um produto (FLAG_MOV); variáveis apenas de M0
(nada que só se sabe depois da oferta); treino jun–out, teste nov–dez (validação temporal).
""")
    cm = D["comparacao_modelos"].copy()
    st.dataframe(cm.style.format("{:.3f}"), width="stretch")
    dec = D["decis"]
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Bar(x=dec["decil"], y=dec["taxa"], marker_color=AZUL, text=[pct(v, 0) for v in dec["taxa"]],
                               textposition="outside", hovertemplate="Decil %{x}: %{y:.1%}<extra></extra>"))
        fig.add_hline(y=(dec["taxa"] * dec["leads"]).sum() / dec["leads"].sum(), line_dash="dash", line_color=TINTA_2,
                      annotation_text="média")
        fig.update_yaxes(tickformat=".0%")
        fig.update_xaxes(title="Decil de escore (1 = maior propensão)", dtick=1)
        fig.update_layout(title="Taxa real por decil – LightGBM, teste nov–dez")
        st.plotly_chart(layout(fig, 380, False), width="stretch")
    with col2:
        fig = go.Figure(go.Scatter(x=[0] + list(dec["decil"] / 10), y=[0] + list(dec["% dos movimentadores acumulado"]),
                                   mode="lines+markers", line=dict(color=AZUL, width=2), name="Modelo"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=CINZA_CLARO, dash="dash"), name="Aleatório"))
        fig.update_xaxes(tickformat=".0%", title="% dos retidos abordados (por ordem de escore)")
        fig.update_yaxes(tickformat=".0%", title="% dos movimentadores capturados")
        fig.update_layout(title="Curva de ganho acumulado")
        st.plotly_chart(layout(fig, 380), width="stretch")
    st.markdown('<p class="nota">AUC na faixa de 0,7 é o esperado para um problema em que a decisão final depende da conversa '
                'e da oferta disponível no momento, informações que não estão na base. O valor do modelo está na ordenação: '
                'o topo da lista concentra bem mais movimentadores que a média.</p>', unsafe_allow_html=True)
