@echo off
REM ==== Rivercast Pipeline ====
REM Stoppe Ausführung bei Fehler
setlocal enabledelayedexpansion

echo [1] ETL: load
python -m src.etl.load || goto :error

echo [2] QC
python -m src.etl.qc || goto :error

echo [3] Features
python -m src.features.build_features || goto :error

echo [4] Training
python -m src.models.train_baseline || goto :error

echo [5] Serving
python -m src.serve.publish || goto :error

echo.
echo Pipeline erfolgreich abgeschlossen!
goto :end

:error
echo.
echo *** FEHLER: Pipeline abgebrochen ***
exit /b 1

:end
endlocal
