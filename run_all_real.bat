@echo off
REM Native Windows runner for the three real-data pipelines.
REM Produces:
REM   risk_report.pdf                  Section 12 verified backtest (Sharpe -0.77)
REM   ml_integration_real.{csv,txt}    Section 10 ML on real data
REM   strategy_variants_real.{csv,txt} Section 8 variants on real data
REM
REM Run from cmd.exe in the repo root:
REM     run_all_real.bat
REM
REM Once the *.txt files exist, paste their contents into Sections 8 and 10
REM of the thesis, then delete the Simulation/Projection Disclosure banners.

echo === Installing dependencies ===
pip install --quiet yfinance pandas-datareader matplotlib scipy statsmodels scikit-learn
if errorlevel 1 goto :error

echo.
echo === [1/3] Section 12: Verified backtest (risk_report.py) ===
python risk_report.py
if errorlevel 1 goto :error

echo.
echo === [2/3] Section 10: ML on real data (ml_integration_real.py) ===
python ml_integration_real.py
if errorlevel 1 goto :error

echo.
echo === [3/3] Section 8: Strategy variants on real data (strategy_variants_real.py) ===
python strategy_variants_real.py
if errorlevel 1 goto :error

echo.
echo === Output files ===
dir risk_report.pdf ml_integration_real.* strategy_variants_real.* 2>nul

echo.
echo Done. Paste *.txt content into thesis Sections 8 and 10, then delete the
echo Simulation/Projection Disclosure banners at the top of those .tex files.
goto :eof

:error
echo.
echo *** A step failed. See output above. ***
exit /b 1
