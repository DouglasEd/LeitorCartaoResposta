@echo off
set VENV_DIR=venv

if exist %VENV_DIR% (
    echo O ambiente virtual '%VENV_DIR%' já existe. Ativando...
) else (
    echo Criando o ambiente virtual...
    python -m venv %VENV_DIR%
)

call %VENV_DIR%\Scripts\activate.bat
