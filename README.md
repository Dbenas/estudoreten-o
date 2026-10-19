# Retenção com movimentação

Case técnico para a vaga de Cientista de Dados: identificar o perfil dos clientes que, ao pedir cancelamento, são retidos **e** alteram algum produto contratado (banda larga, TV, fone ou móvel).

## O que tem aqui

| Caminho | Conteúdo |
|---|---|
| `notebooks/case_retencao.ipynb` | Análise completa: qualidade da base, criação de variáveis, análise descritiva, tipologia das movimentações, modelagem (logística e LightGBM + SHAP), perfis, efeito da ferramenta SOLAR e conclusões |
| `src/preparacao.py` | Leitura, limpeza e criação de variáveis (usado pelo notebook e pelo app) |
| `src/visual.py` | Padrão visual dos gráficos |
| `app/streamlit_app.py` | Dashboard interativo com simulador de atendimento |
| `outputs/figuras/` | Gráficos do notebook em PNG |
| `outputs/app/`, `outputs/modelos/` | Tabelas agregadas e modelos usados pelo dashboard (gerados pelo notebook) |

## Como executar

```bash
pip install -r requirements.txt
```

1. Coloque `bd_case_final_1.xlsx` na raiz do projeto.
2. Execute o notebook `notebooks/case_retencao.ipynb` do início ao fim. A primeira leitura do Excel leva alguns minutos; depois fica em cache em `data/`.
3. Abra o dashboard:

```bash
streamlit run app/streamlit_app.py
```

## Principais resultados

- **Sobrepreço** (quanto o cliente paga acima do preço típico de um produto equivalente na mesma praça) é o principal fator para movimentar: de 21% para ~40% de taxa de troca.
- A movimentação na retenção é, na prática, **migração para o catálogo atual**: só 10% das trocas de BL são downgrade; 87% das trocas de TV vão para streaming.
- **Retenção ≠ receita**: os retidos levam mais receita do funil (13%) que os não retidos (9%), em especial clientes marcados como retidos que zeram a conta em 30 dias.
- **SOLAR**: +8,2 p.p. de retenção comparando clientes equivalentes (escore de propensão), contra +2,5 p.p. na comparação bruta.
- LightGBM com validação temporal (teste nov–dez): AUC 0,69, decil superior com 1,9x a taxa média.

## Observação sobre os dados

A base é da empresa e não deve ser versionada nem publicada. O dashboard usa apenas tabelas agregadas e os modelos treinados.
