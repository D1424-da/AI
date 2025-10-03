# インストールガイド

## システム要件

### 必須環境
- **Python**: 3.8以上（推奨: 3.10, 3.11）
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 20.04+
- **メモリ**: 8GB以上推奨
- **ストレージ**: 2GB以上の空き容量

### GPU使用時の追加要件
- **NVIDIA GPU**: CUDA Compute Capability 3.5以上
- **CUDA**: 11.2以上
- **cuDNN**: 8.1以上

## インストール手順

### 1. リポジトリのクローン
```bash
git clone https://github.com/D1424-da/AI.git
cd AI
```

### 2. 仮想環境の作成
```bash
python -m venv .venv
```

### 3. 仮想環境の有効化

**Windows:**
```powershell
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

### 4. 依存関係のインストール

**CPU版（推奨）:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**GPU版:**
```bash
pip install --upgrade pip
pip install -r requirements-gpu.txt
```

**開発版:**
```bash
pip install --upgrade pip
pip install -r requirements-dev.txt
```

### 5. インストール確認
```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import PIL; print('Pillow:', PIL.__version__)"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
```

### 6. GPU確認（GPU版の場合）
```bash
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

## クイックスタート

### データ構造確認
```bash
python verify_data_structure.py
```

### AI分類アプリ起動
```bash
python pile_classifier_app.py
```

### コマンドライン分類器
```bash
python core_classifier_organizer.py
```

### モデル訓練（メモリ最適化版）
```bash
python train_model_memory_efficient.py
```

## トラブルシューティング

### TensorFlowがインストールできない
```bash
# 最新のpipにアップグレード
pip install --upgrade pip setuptools wheel

# 特定バージョンを指定
pip install tensorflow==2.13.0
```

### GPU認識されない
```bash
# CUDA/cuDNNのバージョン確認
nvidia-smi

# TensorFlow GPU版の再インストール
pip uninstall tensorflow
pip install tensorflow-gpu==2.13.0
```

### メモリ不足エラー
```bash
# config.jsonでバッチサイズを削減
# "batch_size": 32  # デフォルト128から変更
```

### 日本語パスで画像が読めない
→ Pillow (PIL) を使用しているため、この問題は解決済みです。

### 大量画像での重複カウント問題
```bash
# データ構造検証で重複除去確認
python -c "from verify_data_structure import scan_data_structure; scan_data_structure()"
```

## アップデート

### 依存関係の更新
```bash
pip install --upgrade -r requirements.txt
```

### 開発版への更新
```bash
pip install --upgrade -r requirements-dev.txt
```

## アンインストール

```bash
# 仮想環境を無効化
deactivate

# 仮想環境を削除
# Windows
rmdir /s .venv

# Linux/macOS
rm -rf .venv
```

## パフォーマンステスト

### システム情報確認
```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

### 訓練データ統計
```bash
python -c "from verify_data_structure import scan_data_structure; scan_data_structure()"
```

## サポート

- **GitHub Issues**: [https://github.com/D1424-da/AI/issues](https://github.com/D1424-da/AI/issues)
- **技術的質問**: GitHubのDiscussionsページ
- **バグ報告**: GitHub Issues ページ

---

**注意**: 初回セットアップ時は依存関係のダウンロードに時間がかかる場合があります。