@echo off

:: Forçando permissões de ADM no arquivo, pra garantir uma instalação correta dos pacotes
net session >nul 2>&1
if %errorLevel% NEQ 0 (
    echo Solicitando permissao de administrador...
    powershell -Command "Start-Process cmd -ArgumentList '/c cd /d \"%CD%\" && call \"%~f0\"' -Verb RunAs" -Wait
    exit /b
)

:: Forçando a execução na pasta do script
cd /d "%~dp0"

echo Instalando dependencias...
pip install -r requirements.txt
echo Instalacao concluida!
pause
