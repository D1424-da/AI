# ConfigLoader使用例集
# 59,337枚データでの訓練とアプリケーション統合例

from config_loader import ConfigLoader
import numpy as np

def training_script_example():
    """訓練スクリプトでの使用例"""
    print("=== 訓練スクリプトでの統合使用 ===")
    
    # 設定読み込み
    config = ConfigLoader()
    
    # 設定検証
    try:
        config.validate_settings()
        print("✅ 設定検証完了")
    except ValueError as e:
        print(f"❌ 設定エラー: {e}")
        return
    
    # 訓練用設定取得
    trainer_config = config.get_trainer_config()
    
    # 疑似的な訓練クラスの初期化
    print(f"データディレクトリ: {config.data_dir}")
    print(f"画像サイズ: {config.image_size}")
    print(f"バッチサイズ: {config.batch_size}")
    print(f"学習率: {config.learning_rate}")
    print(f"クラス名: {config.class_order}")
    print(f"クラス数: {len(config.class_order)}")
    
    # メモリ最適化設定
    if config.memory_optimization:
        print("🔧 メモリ最適化モードが有効です")
    
    # 早期停止設定
    if config.early_stopping:
        print(f"⏹️ 早期停止: patience={config.patience}")

def application_example():
    """アプリケーションでの使用例"""
    print("\n=== アプリケーションでの統合使用 ===")
    
    # 設定読み込み
    config = ConfigLoader()
    
    # アプリケーション用設定取得
    app_config = config.get_app_config()
    
    # 疑似的なアプリケーションクラスの初期化
    class PileClassifierApp:
        def __init__(self, config_loader):
            self.config = config_loader
            self.class_names = config_loader.class_order
            self.image_size = config_loader.image_size
            self.model_path = config_loader.model_path
            self.code_mapping = config_loader.code_mapping
            self.batch_size = config_loader.batch_size
            
        def load_model(self):
            print(f"📂 モデル読み込み: {self.model_path}")
            print(f"🏷️ クラス数: {len(self.class_names)}")
            return True
            
        def preprocess_image(self, image_path):
            print(f"🖼️ 画像前処理: {self.image_size}")
            # 実際の前処理はここに実装
            return np.zeros((*self.image_size, 3))
            
        def predict_and_rename(self, image_path):
            # 予測とファイル名変更のロジック
            predicted_class = self.class_names[0]  # 疑似予測
            pile_code = self.code_mapping.get(predicted_class, "UNKNOWN")
            print(f"🔍 予測: {predicted_class} -> コード: {pile_code}")
            return pile_code
    
    # アプリケーション初期化・実行
    app = PileClassifierApp(config)
    if app.load_model():
        print("✅ アプリケーション初期化完了")
        app.predict_and_rename("test_image.jpg")

def consistency_check():
    """設定整合性チェック"""
    print("\n=== 設定整合性チェック ===")
    
    config = ConfigLoader()
    
    # クラス順序とコードマッピングの整合性
    class_order = config.class_order
    code_mapping = config.code_mapping
    
    missing_codes = []
    for class_name in class_order:
        if class_name not in code_mapping:
            missing_codes.append(class_name)
    
    if missing_codes:
        print(f"❌ コードマッピング不足: {missing_codes}")
    else:
        print("✅ クラス-コードマッピング整合性OK")
    
    # 重複コードチェック
    codes = list(code_mapping.values())
    if len(codes) != len(set(codes)):
        print("❌ 重複するパイルコードがあります")
    else:
        print("✅ パイルコード重複なし")
    
    # 12クラス完全性チェック
    if len(class_order) == 12 and len(code_mapping) == 12:
        print("✅ 12クラス完全対応")
    else:
        print(f"❌ クラス数不整合: クラス順序={len(class_order)}, コードマッピング={len(code_mapping)}")

def memory_usage_example():
    """メモリ使用量管理例"""
    print("\n=== メモリ使用量管理 ===")
    
    config = ConfigLoader()
    
    print(f"最大メモリ使用率: {config.max_memory_usage:.1%}")
    print(f"推奨バッチサイズ: {config.batch_size}")
    
    # 59,337枚での推定メモリ使用量計算
    image_size = config.image_size
    total_images = 59337
    bytes_per_image = image_size[0] * image_size[1] * 3 * 4  # float32
    batch_memory_mb = (config.batch_size * bytes_per_image) / (1024 * 1024)
    
    print(f"📊 画像1枚あたりメモリ: {bytes_per_image / 1024:.1f}KB")
    print(f"📊 バッチメモリ使用量: {batch_memory_mb:.1f}MB")
    print(f"📊 総データセット: {total_images:,}枚")
    
    # バッチ数計算
    total_batches = (total_images + config.batch_size - 1) // config.batch_size
    print(f"📊 総バッチ数: {total_batches:,}")

if __name__ == "__main__":
    print("=== ConfigLoader 統合使用例 ===")
    print("59,337枚データセットでの訓練・アプリケーション統合")
    
    training_script_example()
    application_example()
    consistency_check()
    memory_usage_example()
    
    print("\n" + "="*60)
    print("🎉 ConfigLoaderによる統合が完了しました！")
    print("これで訓練スクリプトとアプリケーション間の")
    print("設定整合性が大幅に向上し、システム全体の")
    print("信頼性が確保されます。")
    print("="*60)