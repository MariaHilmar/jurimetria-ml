# ADR 001 - Dataset sintético e recorte público

**Status:** aceito  
**Data:** 2026-08-13

## Contexto

Criar um repositório público com o ciclo clássico de ML (treino, validação, predição e métricas) semelhante ao que implementei no sistema "sj-pipeline-bi".

## Decisão

1. Gerar um dataset **sintético** com processo generativo conhecido (`ml/dataset.py`).
2. Manter o ciclo completo e pequeno: Pandas/NumPy, sklearn, XGBoost, calibração, F1/AUC/Brier, FastAPI.
3. Não copiar módulos, tabelas, prompts nem features do produto privado.
4. Documentar no README que este repositório **não** prevê desfecho judicial real.

## Consequências

- O modelo aprende um sinal artificial. AUC alto aqui não é performance em tribunal.
- A API e os testes são reproduzíveis com seed fixa, sem credenciais.
- JurisSync permanece a evidência de jurimetria descritiva (SQL). Este repo é a evidência preditiva.
