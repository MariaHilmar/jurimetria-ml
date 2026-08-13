# ADR 001 - Dataset sintético e recorte público

**Status:** aceito  
**Data:** 2026-08-13

## Contexto

O ciclo clássico de ML (treino, validação, predição e métricas) existe em um pipeline privado. Publicar esse código inteiro misturaria produto, schema e risco de dados. Um clone também seria difícil de revisar: volume alto, dependências de banco e regras de negócio.

O portfólio público precisava de um recorte que um recrutador consiga clonar, treinar e chamar `/predict` em minutos.

## Decisão

1. Gerar um dataset **sintético** com processo generativo conhecido (`ml/dataset.py`).
2. Manter o ciclo completo e pequeno: Pandas/NumPy, sklearn, XGBoost, calibração, F1/AUC/Brier, FastAPI.
3. Não copiar módulos, tabelas, prompts nem features do produto privado.
4. Documentar no README que este repositório **não** prevê desfecho judicial real.

## Consequências

- O modelo aprende um sinal artificial. AUC alto aqui não é performance em tribunal.
- A API e os testes são reproduzíveis com seed fixa, sem credenciais.
- JurisSync permanece a evidência de jurimetria descritiva (SQL). Este repo é a evidência preditiva.
