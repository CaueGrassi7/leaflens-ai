#!/usr/bin/env python3
"""
Script alternativo para iniciar o servidor quando estiver dentro de backend/.
Execute: python run.py ou python backend/run.py
"""
import sys
from pathlib import Path

# Adicionar a raiz do projeto ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(project_root / "backend")]
    )

