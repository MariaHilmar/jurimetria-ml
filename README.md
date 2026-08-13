# Jurimetria ML

Recorte público do **ciclo clássico de Machine Learning**: gerar dados, treinar, avaliar e servir predição via API.

Não é um modelo de desfecho judicial real. Não é o pipeline de produção do Situação Jurídica. O dataset é **sintético**, com um processo generativo conhecido, para o recrutador conseguir clicar, treinar e inspecionar métricas sem dados sigilosos.

O [JurisSync](https://github.com/MariaHilmar/juris-sync) cobre jurimetria **descritiva** (agregações SQL). Este repositório cobre a fatia **preditiva** (sklearn + XGBoost + FastAPI).

[![CI](https://github.com/MariaHilmar/jurimetria-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/MariaHilmar/jurimetria-ml/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![XGBoost](https://img.shields.io/badge/XGBoost-sklearn_pipeline-orange)

## O que este repo prova

```text
dataset sintético -> split estratificado -> pipeline sklearn
  -> XGBoost calibrado -> F1 / AUC / Brier
  -> artefato joblib -> API /predict e /metrics
```

| Peça | Onde |
|------|------|
| Coleta / features | `ml/dataset.py`, `ml/features.py` |
| Treino e validação | `ml/train.py` |
| Predição | `POST /predict` |
| Monitoramento do modelo | `GET /metrics`, `GET /health` |
| Testes | `tests/` (dataset, treino, API) |

## Fora de escopo (de propósito)

- Dados reais de processos, tribunais ou escritórios
- Coleta Escavador / DataJud
- NLP, OCR, dbt, multi-tenant, model registry de produto
- TensorFlow, Keras, PyTorch, Django

Essas peças, quando existem, ficam no produto privado. Aqui o objetivo é um recorte **legível em uma revisão de PR**.

## Stack

| Camada | Tecnologia |
|--------|------------|
| Dados | Pandas, NumPy |
| Modelo | scikit-learn (pipeline, calibração) + XGBoost |
| API | FastAPI + Pydantic v2 |
| Artefato | joblib |
| Qualidade | pytest, Ruff, Black, Mypy |
| Runtime | Python 3.12, Docker opcional |

## Como rodar

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

python -m ml.train
uvicorn app.main:app --reload
```

- OpenAPI: http://127.0.0.1:8000/docs
- Saúde: `GET /health`
- Métricas do último treino: `GET /metrics`

### Exemplo de predição

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body '{
  "tribunal": "TJSP",
  "area_direito": "consumidor",
  "grau": 1,
  "valor_causa": 25000,
  "qtd_movimentacoes": 14,
  "dias_tramitacao": 420,
  "tem_liminar": true
}'
```

Resposta típica:

```json
{
  "probabilidade_procedente": 0.72,
  "classe": "procedente",
  "limiar": 0.5,
  "versao_modelo": "xgb-calibrated-42-2000"
}
```

## Testes

```powershell
python -m pytest
ruff check app ml tests
black --check app ml tests
mypy app ml
```

A suíte treina um modelo pequeno em diretório temporário. Não depende de artefato commitado.

## Docker

```powershell
docker build -t jurimetria-ml .
docker run --rm -p 8000:8000 jurimetria-ml
```

A imagem treina o modelo no build (dataset sintético, seed fixa) para o container já subir com `/predict` pronto.

## Decisão de desenho

O dataset sintético e o recorte deliberado estão em [`docs/adr/001-dataset-sintetico.md`](docs/adr/001-dataset-sintetico.md).

## Licença

MIT. Ver [LICENSE](LICENSE).
