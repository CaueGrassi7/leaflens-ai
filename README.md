# LeafLens AI

Sistema de detecção de doenças em folhas de tomate usando Machine Learning.

## Estrutura do Projeto

```
leaflens-ai/
├── backend/          # API FastAPI
├── frontend/         # Aplicação Next.js
└── ml/              # Modelos e notebooks de ML
```

## Como Executar o Backend

### Opção 1: Script Python (Recomendado)
```bash
# Da raiz do projeto
python run_backend.py
```

### Opção 2: Uvicorn direto (da raiz)
```bash
# IMPORTANTE: Execute da raiz do projeto, não de dentro de backend/
cd /Users/cauegrassi/Dev/leaflens-ai
source venv/bin/activate
uvicorn backend.app.main:app --reload
```

### Opção 3: Se estiver dentro de backend/
```bash
# De dentro de backend/, use:
python run.py
# ou
python ../run_backend.py
```

## Ambiente Virtual

```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual
source venv/bin/activate  # macOS/Linux
# ou
venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt
```

## Endpoints da API

- `GET /health` - Health check
- `POST /predict` - Predição de doença em imagem de folha

## Desenvolvimento

O servidor roda em `http://localhost:8000` por padrão.


