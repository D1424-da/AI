@echo off
chcp 65001 >nul
echo ====================================
echo   杭種コード自動追加アプリ v1.0
echo ====================================
echo.

REM Pythonの確認
python --version >nul 2>&1
if errorlevel 1 (
    echo エラー: Pythonがインストールされていません。
    echo Python 3.8以上をインストールしてください。
    pause
    exit /b 1
)

REM 依存関係の確認とインストール
echo 依存関係を確認中...
pip show pillow >nul 2>&1
if errorlevel 1 (
    echo 必要なライブラリをインストール中...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo エラー: ライブラリのインストールに失敗しました。
        pause
        exit /b 1
    )
)

REM アプリケーション起動
echo.
echo アプリケーションを起動中...
python pile_classifier_app.py

if errorlevel 1 (
    echo.
    echo エラーが発生しました。
    pause
)
