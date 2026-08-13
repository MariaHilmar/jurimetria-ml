# Contribuindo

Este repositório é um recorte de portfólio. Mudanças devem preservar o ciclo curto (dataset -> treino -> API) e a honestidade do dataset sintético.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

## Checks locais

```powershell
ruff check app ml tests
black app ml tests
mypy app ml
python -m pytest
```

PRs entram em `feat/**` contra `main`. O CI bloqueia merge se lint, tipos ou testes falharem.
