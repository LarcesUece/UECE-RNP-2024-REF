@echo off
REM Script de Instalação Automática para GPU
REM Autor: Sistema de IA
REM Data: 2025-07-23

echo ================================================
echo    INSTALACAO AUTOMATICA - TENSORFLOW GPU
echo ================================================
echo.

REM Ativa ambiente virtual se existir
if exist ".venv\Scripts\activate.bat" (
    echo  Ativando ambiente virtual...
    call .venv\Scripts\activate.bat
) else (
    echo Ambiente virtual não detectado
)

echo.
echo Instalação básica (sem conflitos)
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels jupyter
if %errorlevel% neq 0 (
    echo  Erro na instalação básica
    pause
    exit /b 1
)

echo.
echo Bibliotecas básicas instaladas!
echo.

echo  TensorFlow com GPU (automática) 
pip install tensorflow[and-cuda]
if %errorlevel% equ 0 (
    echo TensorFlow GPU instalado com sucesso!
    goto test_installation
)

echo.
echo   Falha na instalação automática. Tentando versão específica...
echo.

echo TensorFlow versão estável
echo ================================================
pip install tensorflow==2.15.0
if %errorlevel% equ 0 (
    echo TensorFlow  instalado!
    goto test_installation
)

echo.
echo  TensorFlow CPU apenas
echo ================================================
pip install tensorflow
if %errorlevel% equ 0 (
    echo TensorFlow CPU instalado!
    echo Modelos LSTM rodarão na CPU (mais lento)
    goto test_installation
)

echo.
echo ERRO: Não foi possível instalar TensorFlow
echo Tente instalar manualmente:
echo  pip install tensorflow
pause
exit /b 1

:test_installation
echo.
echo 🧪 TESTANDO INSTALAÇÃO
echo ================================================
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPU disponível:', len(tf.config.list_physical_devices('GPU')) > 0)"
if %errorlevel% neq 0 (
    echo  Erro no teste do TensorFlow
    pause
    exit /b 1
)

echo.
echo INSTALANDO DEPENDÊNCIAS EXTRAS
echo ================================================
pip install xgboost psutil
if %errorlevel% neq 0 (
    echo  Algumas dependências extras falharam (não crítico)
)

echo.
echo  INSTALAÇÃO CONCLUÍDA!
echo ================================================
echo    Agora você pode executar:
echo    python test_gpu.py              (testar GPU)
echo    python time_series_prediction_model.py  (modelo completo)
echo.
pause
