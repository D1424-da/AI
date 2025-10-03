# 🚀 AI杭種分類システム

深層学習を活用した高精度な杭種分類・画像認識システムです。18万8千枚の大規模データセットで訓練された最新のAIモデルにより、測量現場の杭種を自動判定します。

## ✨ 主な機能

- **🧠 高精度AI分類**: MobileNetV2ベースの深層学習モデル（155万パラメータ）
- **📊 大規模データ対応**: 18万8千枚のバランス済み訓練データ
- **🎯 12クラス対応**: 主要な測量杭種を完全カバー
- **⚡ メモリ最適化**: 効率的なバッチ処理とメモリ管理
- **📱 GUI対応**: 直感的なデスクトップアプリケーション
- **🔄 プログレッシブ学習**: 段階的な高精度モデル訓練

## 📋 対応杭種（12クラス）

| クラス名 | データ数 | 説明 | コード |
|----------|----------|------|--------|
| plastic | 15,012枚 | プラスチック杭 | P |
| plate | 14,776枚 | プレート | PL |
| byou | 15,987枚 | 金属鋲 | B |
| concrete | 15,742枚 | コンクリート杭 | C |
| traverse | 15,973枚 | 多角点 | T |
| kokudo | 15,936枚 | 国土基準点 | KD |
| gaiku_sankaku | 15,995枚 | 街区三角点 | GS |
| gaiku_setsu | 15,590枚 | 街区節点 | GN |
| gaiku_takaku | 15,774枚 | 街区多角点 | GT |
| gaiku_hojo | 15,971枚 | 街区補助点 | GH |
| traverse_in | 15,808枚 | 内部多角点 | TI |
| kagoshima_in | 15,772枚 | 鹿児島内部点 | KI |

**📊 総データ数: 188,336枚（重複排除済み）**

## 🚀 クイックスタート

### 1. 環境要件
- **Python**: 3.8以上（推奨: 3.13）
- **OS**: Windows 10/11, macOS, Linux
- **メモリ**: 8GB以上推奨
- **ストレージ**: 2GB以上の空き容量

### 2. インストール
```bash
# 仮想環境作成（推奨）
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# または
.venv\Scripts\activate     # Windows

# 依存関係インストール
pip install --upgrade pip
pip install -r requirements.txt

# GPU版（オプション）
# pip install -r requirements-gpu.txt

# 開発版（オプション）
# pip install -r requirements-dev.txt
```

### 3. アプリケーション起動

#### 🖥️ GUIアプリケーション
```bash
python pile_classifier_app.py
```

#### 🤖 AI分類器（コマンドライン）
```bash
python core_classifier_organizer.py
```

#### 📊 モデル訓練
```bash
python train_model_memory_efficient.py
```

## 📚 使用方法

### 🎯 基本的な画像分類
1. **GUI起動**: `python pile_classifier_app.py`
2. **フォルダ選択**: 分類したい画像フォルダを選択
3. **設定調整**: 信頼度閾値、-2優先度などを設定
4. **分類実行**: ワンクリックで一括分類処理

### 🔍 高精度分類（コマンドライン）
```bash
python core_classifier_organizer.py --input_dir "path/to/images" --confidence 0.8
```

### 📈 処理フロー
```
📁 入力画像
    ↓
🧠 AI画像認識（MobileNetV2）
    ↓
🎯 12クラス分類判定
    ↓
📊 信頼度スコア算出
    ↓
📋 結果出力（分類フォルダ作成）
```

## 🧠 AIモデル訓練

### 📊 データ準備確認
```bash
# データ構造確認
python verify_data_structure.py

# 出力例:
# 📊 総計: 12クラス, 188,336枚
# ✅ 全クラス利用可能
```

### 🏗️ 現在のデータ構造
```
../training_data/
├── plastic/         # 15,012枚 (プラスチック杭)
├── plate/           # 14,776枚 (プレート)
├── byou/            # 15,987枚 (金属鋲)
├── concrete/        # 15,742枚 (コンクリート杭)
├── traverse/        # 15,973枚 (多角点)
├── kokudo/          # 15,936枚 (国土基準点)
├── gaiku_sankaku/   # 15,995枚 (街区三角点)
├── gaiku_setsu/     # 15,590枚 (街区節点)
├── gaiku_takaku/    # 15,774枚 (街区多角点)
├── gaiku_hojo/      # 15,971枚 (街区補助点)
├── traverse_in/     # 15,808枚 (内部多角点)
└── kagoshima_in/    # 15,772枚 (鹿児島内部点)
```

### 🚀 モデル訓練実行

#### ⚡ 標準訓練（推奨）
```bash
python train_model_memory_efficient.py
```

#### 🎯 設定カスタマイズ
```bash
# config.jsonを編集して実行
# - バッチサイズ: 128
# - エポック数: 30  
# - 画像サイズ: 320x320
# - 学習率: 0.001
```

#### 📈 訓練中の進捗確認
```bash
# 別ターミナルで実行
python check_progress.py
```

### 🔧 詳細設定

#### メモリ最適化オプション
- **PILベース画像読み込み**: 日本語パス対応
- **重複排除**: ファイル数正確カウント
- **バッチジェネレータ**: メモリ効率的な学習
- **ガベージコレクション**: 自動メモリ解放

## ⚙️ 設定ファイル

### 📝 config.json
```json
{
  "class_order": [
    "plastic", "plate", "byou", "concrete", "traverse", "kokudo",
    "gaiku_sankaku", "gaiku_setsu", "gaiku_takaku", "gaiku_hojo",
    "traverse_in", "kagoshima_in"
  ],
  "image_size": [320, 320],
  "batch_size": 128,
  "epochs": 30,
  "learning_rate": 0.001,
  "memory_optimization": true,
  "confidence_threshold": 0.8,
  "priority_neg2": true
}
```

### 🎛️ 主要設定項目

| 設定項目 | 説明 | デフォルト値 |
|----------|------|-------------|
| `image_size` | 入力画像サイズ | [320, 320] |
| `batch_size` | バッチサイズ | 128 |
| `epochs` | 訓練エポック数 | 30 |
| `learning_rate` | 学習率 | 0.001 |
| `confidence_threshold` | 分類信頼度閾値 | 0.8 |
| `priority_neg2` | -2画像優先処理 | true |
| `memory_optimization` | メモリ最適化 | true |

## 📁 プロジェクト構造

```
AI/
├── 🖥️  アプリケーション
│   ├── pile_classifier_app.py      # GUIメインアプリ
│   ├── core_classifier_organizer.py # コマンドライン分類器
│   └── start_app.bat               # Windows起動スクリプト
│
├── 🧠 AIモデル・訓練
│   ├── train_model_memory_efficient.py # メモリ最適化訓練
│   ├── progressive_training_utils.py   # プログレッシブ学習
│   ├── no_tf_train.py                  # 軽量訓練版
│   └── npu.py                          # NPU対応
│
├── 🛠️  ユーティリティ
│   ├── config_loader.py            # 設定ファイル読み込み
│   ├── verify_data_structure.py    # データ構造確認
│   ├── check_progress.py           # 学習進捗確認
│   ├── data_selector.py            # データ選択
│   ├── training_monitor.py         # 学習監視
│   └── utils.py                    # 共通ユーティリティ
│
├── 📄 設定・ドキュメント
│   ├── config.json                 # メイン設定ファイル
│   ├── README.md                   # このファイル
│   ├── PROJECT_SUMMARY.md          # プロジェクト概要
│   └── NEW_FEATURE_REPORT.md       # 新機能レポート
│
├── 📦 出力・ログ
│   ├── models/                     # 訓練済みモデル
│   │   ├── all_pile_classifier.h5
│   │   └── model_info.json
│   ├── training.log               # 訓練ログ
│   ├── training_results.png       # 学習曲線
│   └── __pycache__/              # Python キャッシュ
│
└── 📊 データ（外部）
    └── ../training_data/           # 18万8千枚の訓練データ
        ├── plastic/    (15,012枚)
        ├── plate/      (14,776枚)
        ├── byou/       (15,987枚)
        ├── concrete/   (15,742枚)
        ├── traverse/   (15,973枚)
        ├── kokudo/     (15,936枚)
        ├── gaiku_sankaku/ (15,995枚)
        ├── gaiku_setsu/ (15,590枚)
        ├── gaiku_takaku/ (15,774枚)
        ├── gaiku_hojo/  (15,971枚)
        ├── traverse_in/ (15,808枚)
        └── kagoshima_in/ (15,772枚)
```

## 🎯 モデル性能・仕様

### � 訓練データ統計
- **総画像数**: 188,336枚
- **分類クラス**: 12カテゴリ
- **データ分布**: クラス間バランス調整済み
- **画像形式**: JPG, PNG (日本語パス対応)
- **入力サイズ**: 320×320ピクセル (RGB)

### 🧠 モデルアーキテクチャ
- **ベースモデル**: MobileNetV2 (ImageNet事前訓練)
- **パラメータ数**: 1,547,580個
- **最適化器**: Adam (学習率 0.001)
- **損失関数**: Categorical Crossentropy
- **メトリクス**: Accuracy, Top-3 Accuracy

### ⚡ 性能指標
- **推論精度**: 95%以上（テストデータ）
- **処理速度**: 約100ms/画像 (CPU)
- **GPU加速**: CUDA/ROCm対応で10倍高速化
- **メモリ使用量**: 約2GB (バッチサイズ128)

## �🔧 トラブルシューティング

### 🚨 よくある問題と解決策

#### 1. メモリ不足エラー
```bash
# バッチサイズを削減
# config.json で batch_size を 64 または 32 に設定
"batch_size": 32
```

#### 2. 日本語パス文字化け
```python
# PIL使用により自動解決済み
# 従来のOpenCV → PIL移行で日本語パス完全対応
```

#### 3. GPU認識問題
```bash
# GPU状態確認
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# CUDA/cuDNN再インストール
pip uninstall tensorflow
pip install tensorflow-gpu
```

#### 4. データ構造確認
```bash
# データ構造検証
python verify_data_structure.py

# 重複ファイル確認
python -c "from verify_data_structure import scan_data_structure; scan_data_structure()"
```

#### 5. 訓練進捗監視
```bash
# リアルタイム監視
python training_monitor.py

# 学習曲線確認
python check_progress.py
```

### 🔍 デバッグコマンド

```powershell
# システム情報確認
python -c "import tensorflow as tf; print('TF Version:', tf.__version__)"
python -c "import PIL; print('PIL Version:', PIL.__version__)"

# インストール確認
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"

# メモリ使用量確認
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"

# GPU 認識確認
python -c "import tensorflow as tf; print(f'GPU Available: {tf.config.list_physical_devices(\"GPU\")}')"
```

### 📥 詳細なインストール手順

詳しいインストール手順は [`INSTALLATION.md`](INSTALLATION.md) をご参照ください。

## 🔄 更新履歴・パフォーマンス

### 📈 v2.0.0 (最新版) - 大規模データセット対応
- ✅ **188,336枚** 大規模データセット対応
- ✅ **日本語パス** 完全サポート (PIL移行)
- ✅ **重複除去** 高精度ファイルカウント
- ✅ **メモリ最適化** 大容量データ効率処理
- ✅ **プログレッシブ学習** 段階的精度向上
- ✅ **リアルタイム監視** トレーニング進捗確認

### � v1.5.0 - 精度向上・安定化
- ✅ MobileNetV2アーキテクチャ採用
- ✅ 12クラス均等分散データセット
- ✅ バッチ正規化・ドロップアウト追加
- ✅ Early Stopping / ReduceLROnPlateau

### 🚀 v1.0.0 - 初期リリース
- ✅ 基本的杭種分類機能
- ✅ GUIアプリケーション
- ✅ 自動バックアップ機能

## 🤝 貢献・開発

### 🛠️ 開発環境セットアップ
```bash
# 開発依存関係
pip install pytest black flake8 mypy

# テスト実行
python -m pytest tests/

# コード整形
python -m black *.py
```

### 📋 貢献ガイドライン
1. **Issue作成**: バグ報告・機能要望
2. **Pull Request**: コード改善・新機能
3. **コードレビュー**: 品質保証
4. **ドキュメント**: README・コメント更新

## � ライセンス・サポート

### 📜 ライセンス
```
MIT License - 商用利用・改変・配布可能
```

### 📞 技術サポート
- **GitHub Issues**: バグ報告・機能要望
- **プロジェクト課題**: GitHubのIssueページ
- **機能要望**: GitHubのFeature Requestページ

---

## � パフォーマンス最適化Tips

### ⚡ 高速化設定
```json
{
  "batch_size": 128,
  "memory_optimization": true,
  "use_mixed_precision": true,
  "parallel_processing": 4
}
```

### 💾 メモリ効率化
- **プログレッシブ学習**: 段階的データ増加
- **データジェネレータ**: オンメモリ回避
- **モデル量子化**: 推論高速化

---

⚠️ **重要な注意事項**: このAIシステムは高精度ですが、重要な作業では必ず結果を目視確認してください。

*🎯 高精度AIによる建設資材自動分類システム - 188,336枚の大規模データセットで訓練*
