# ADR 002 - Flask, Keras e Kubernetes como extras

**Status:** aceito  
**Data:** 2026-08-13

## Contexto

O nucleo do recorte ja prova sklearn + XGBoost + FastAPI. Algumas vagas pedem Flask, TensorFlow/Keras e Kubernetes. Colocar isso no caminho feliz do treino deixaria o CI pesado.

## Decisao

1. Flask reusa ModelService (`app/flask_app.py`). FastAPI permanece a API principal.
2. Keras/TensorFlow ficam em `requirements-dl.txt` e `ml/train_keras.py` (MLP de 1 epoca em CPU).
3. Kubernetes fica em YAML versionado (`k8s/`), validado por kubeconform, sem cluster obrigatorio.

## Consequencias

- O ATS encontra Flask, Keras e Deployment no mesmo repositorio.
- Quem clona so `requirements-dev.txt` continua com o ciclo XGBoost leve.
