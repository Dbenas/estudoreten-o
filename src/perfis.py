"""Transforma as folhas de uma árvore de decisão em regras legíveis."""
import numpy as np

BINARIAS = {"tem_tv": "TV", "tem_fone": "Fone", "flag_movel": "móvel na operadora",
            "plano_de_retencao_m0": "plano RET", "plano_old_m0": "plano OLD",
            "origem_solar": "SOLAR", "rede_gpon": "rede GPON"}
CONTINUAS = {"receita_m0": ("ticket", "R$ "), "VALOR_TV_M0": ("valor TV", "R$ "),
             "velocidade_bl_m0": ("velocidade", "")}


def regras_folhas(modelo, nomes):
    """Lista de condições (variável, operador, limite) que leva a cada folha."""
    arv = modelo.tree_
    regras = {}

    def descer(no, cond):
        if arv.children_left[no] == -1:
            regras[no] = cond
            return
        var, lim = nomes[arv.feature[no]], arv.threshold[no]
        descer(arv.children_left[no], cond + [(var, "<=", lim)])
        descer(arv.children_right[no], cond + [(var, ">", lim)])

    descer(0, [])
    return regras


def _intervalo(nome, lo, hi, pre=""):
    if lo > -np.inf and hi < np.inf:
        return f"{nome} entre {pre}{lo:,.0f} e {pre}{hi:,.0f}"
    if lo > -np.inf:
        return f"{nome} > {pre}{lo:,.0f}"
    return f"{nome} ≤ {pre}{hi:,.0f}"


def texto_regra(cond):
    """Junta condições repetidas da mesma variável e escreve a regra em português."""
    lim = {}
    for var, op, v in cond:
        lo, hi = lim.get(var, (-np.inf, np.inf))
        lim[var] = (lo, min(hi, v)) if op == "<=" else (max(lo, v), hi)

    partes = []
    for var, (lo, hi) in lim.items():
        if var in BINARIAS:
            partes.append(("com " if lo >= 0.5 else "sem ") + BINARIAS[var])
        elif var == "tem_sobrepreco_bl":
            # quem não tem BL entra com -100% (fillna(-1)), então cai sempre no lado "≤"
            if lo > -np.inf and hi < np.inf:
                partes.append(f"BL entre {lo:+.0%} e {hi:+.0%} do preço típico")
            elif lo > -np.inf:
                partes.append(f"BL acima de {lo:+.0%} do preço típico")
            else:
                partes.append(f"BL até {hi:+.0%} do preço típico (ou sem BL)")
        elif var in ("qtd_produtos", "n_leads_anteriores"):
            nome = "produtos" if var == "qtd_produtos" else "leads anteriores"
            a = int(np.floor(lo)) + 1 if lo > -np.inf else None
            b = int(np.floor(hi)) if hi < np.inf else None
            if a is not None and b is not None:
                partes.append(f"{a} {nome}" if a == b else f"{a} a {b} {nome}")
            elif a is not None:
                partes.append(f"{a}+ {nome}")
            elif b == 1 and var == "qtd_produtos":
                partes.append("só 1 produto")
            else:
                partes.append(f"até {b} {nome}")
        else:
            nome, pre = CONTINUAS.get(var, (var, ""))
            partes.append(_intervalo(nome, lo, hi, pre))
    return " · ".join(partes)
