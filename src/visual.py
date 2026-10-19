"""Padrão visual dos gráficos do notebook (matplotlib)."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

# Paleta categórica em ordem fixa (validada para daltonismo nos pares adjacentes)
CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
AZUL, LARANJA, VERDE_AGUA, AMARELO = CORES[:4]
CINZA = "#898781"
CINZA_CLARO = "#c3c2b7"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
GRADE = "#e1e0d9"
FUNDO = "#fcfcfb"
# Divergente azul <-> vermelho com meio neutro
DIVERGENTE = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f4a7a6", "#e34948", "#a32a2a"]
SEQUENCIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def aplicar_estilo():
    mpl.rcParams.update({
        "figure.facecolor": FUNDO,
        "axes.facecolor": FUNDO,
        "savefig.facecolor": FUNDO,
        "figure.dpi": 110,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelcolor": TINTA_2,
        "axes.edgecolor": CINZA_CLARO,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRADE,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": CINZA,
        "ytick.color": CINZA,
        "text.color": TINTA,
        "axes.prop_cycle": mpl.cycler(color=CORES),
        "legend.frameon": False,
        "lines.linewidth": 2,
    })


fmt_pct = PercentFormatter(1.0, decimals=0)
fmt_reais = FuncFormatter(lambda v, _: f"R$ {v:,.0f}".replace(",", "."))
fmt_milhar = FuncFormatter(lambda v, _: f"{v/1000:,.0f} mil".replace(",", "."))


def subtitulo(ax, texto):
    """Linha de apoio abaixo do título, em cinza."""
    ax.text(0, 1.02, texto, transform=ax.transAxes, fontsize=9, color=TINTA_2, va="bottom")


def rotular_barras(ax, barras, fmt="{:.0%}", dentro=False, cor=TINTA_2, tamanho=8.5):
    for b in barras:
        if getattr(ax, "_horizontal", False):
            val = b.get_width()
            ax.text(val, b.get_y() + b.get_height() / 2, " " + fmt.format(val),
                    va="center", ha="left", fontsize=tamanho, color=cor)
        else:
            val = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, val, fmt.format(val),
                    va="bottom", ha="center", fontsize=tamanho, color=cor)


def barras_h(ax, rotulos, valores, cor=AZUL, fmt="{:.0%}", destaque=None):
    """Barra horizontal simples com rótulo no fim de cada barra."""
    cores = [cor] * len(valores)
    if destaque is not None:
        cores = [cor if i in destaque else CINZA_CLARO for i in range(len(valores))]
    bars = ax.barh(rotulos, valores, color=cores, height=0.7)
    ax._horizontal = True
    rotular_barras(ax, bars, fmt=fmt)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.invert_yaxis()
    return bars
