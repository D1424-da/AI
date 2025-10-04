# 設定ファイルローダー（12クラス・大容量データセット対応）
import json
from pathlib import Path

class ConfigLoader:
    """設定ファイル読み込み・管理クラス（統合システム対応）"""
    
    def __init__(self, config_path="config.json", validate=True):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # 初期化時に検証実行
        if validate:
            try:
                self.validate_settings()
            except ValueError as e:
                print(f"警告: 設定検証エラー - {e}")
                print("デフォルト設定にフォールバック中...")
                self.config = self._get_default_config()
    
    def _load_config(self):
        """設定ファイル読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"設定ファイル読み込みエラー: {e}")
            return self._get_default_config()
    
    def _get_default_config(self):
        """デフォルト設定（12クラス完全対応）"""
        return {
            "app_info": {
                "version": "2.0.0",
                "model_type": "12_class_pile_classifier",
                "last_updated": "2024-12-27"
            },
            "model_settings": {
                "model_path": "models/all_pile_classifier.h5",
                "model_info_path": "models/all_pile_model_info.json",
                "image_size": [224, 224],
                "batch_size": 32,
                "learning_rate": 0.001,
                "epochs": 50,
                "use_core_only": False,
                "input_shape": [224, 224, 3],
                "num_classes": 12
            },
            "pile_codes": {
                "class_order": [
                    "plastic", "plate", "byou", "concrete",
                    "traverse", "kokudo", "gaiku_sankaku", "gaiku_setsu", 
                    "gaiku_takaku", "gaiku_hojo", "traverse_in", "kagoshima_in"
                ],
                "code_mapping": {
                    "plastic": "P", "plate": "PL", "byou": "B", "concrete": "C",
                    "traverse": "T", "kokudo": "KD", "gaiku_sankaku": "GS", 
                    "gaiku_setsu": "GT", "gaiku_takaku": "GH", "gaiku_hojo": "GK", 
                    "traverse_in": "TI", "kagoshima_in": "KI"
                }
            },
            "training_settings": {
                "data_dir": "training_data",
                "memory_optimization": True,
                "validation_split": 0.2,
                "early_stopping": True,
                "patience": 10,
                "reduce_lr_on_plateau": True
            },
            "processing_settings": {
                "use_12_classes": True,
                "max_memory_usage": 0.8,
                "temp_dir": "temp",
                "cleanup_temp": True
            }
        }
    
    @property
    def model_path(self):
        """モデルパス取得"""
        return self.config.get("model_settings", {}).get("model_path", "models/all_pile_classifier.h5")
    
    @property
    def model_info_path(self):
        """モデル情報パス取得"""
        return self.config.get("model_settings", {}).get("model_info_path", "models/all_pile_model_info.json")
    
    @property
    def num_classes(self):
        """クラス数取得"""
        return self.config.get("model_settings", {}).get("num_classes", 12)
    
    @property
    def input_shape(self):
        """入力形状取得"""
        return tuple(self.config.get("model_settings", {}).get("input_shape", [224, 224, 3]))
    
    @property
    def code_mapping(self):
        """コードマッピング取得（従来形式対応）"""
        # 従来のconfig.jsonの階層構造に対応
        core_classes = self.config.get("pile_codes", {}).get("core_classes", {})
        extended_classes = self.config.get("pile_codes", {}).get("extended_classes", {})
        
        # 統合マッピング作成
        mapping = {}
        mapping.update(core_classes)
        mapping.update(extended_classes)
        
        # デフォルト形式もチェック
        default_mapping = self.config.get("pile_codes", {}).get("code_mapping", {})
        if default_mapping:
            mapping.update(default_mapping)
        
        return mapping
    
    @property
    def data_dir(self):
        """データディレクトリ取得"""
        return self.config.get("training_settings", {}).get("data_dir", "training_data")
    
    @property
    def validation_split(self):
        """バリデーション分割率取得"""
        return self.config.get("training_settings", {}).get("validation_split", 0.2)
    
    @property
    def early_stopping(self):
        """早期停止フラグ取得"""
        return self.config.get("training_settings", {}).get("early_stopping", True)
    
    @property
    def patience(self):
        """早期停止耐性取得"""
        return self.config.get("training_settings", {}).get("patience", 10)
    
    @property
    def max_memory_usage(self):
        """最大メモリ使用率取得"""
        return self.config.get("processing_settings", {}).get("max_memory_usage", 0.8)
    
    @property
    def class_order(self):
        """クラス順序取得"""
        # training_settings から取得、フォールバックでpile_codes.class_order、最終フォールバックでデフォルト
        training_order = self.config.get("training_settings", {}).get("class_order", [])
        if training_order:
            return training_order
        
        pile_order = self.config.get("pile_codes", {}).get("class_order", [])
        if pile_order:
            return pile_order
        
        # デフォルト12クラス順序
        return [
            "plastic", "plate", "byou", "concrete", "traverse", "kokudo",
            "gaiku_sankaku", "gaiku_setsu", "gaiku_takaku", "gaiku_hojo", 
            "traverse_in", "kagoshima_in"
        ]
    
    @property
    def image_size(self):
        """画像サイズ取得"""
        # training_settings を優先、フォールバックでmodel_settings
        training_size = self.config.get("training_settings", {}).get("image_size", [])
        if training_size:
            return tuple(training_size)
        
        model_size = self.config.get("model_settings", {}).get("image_size", [224, 224])
        return tuple(model_size)
    
    @property
    def batch_size(self):
        """バッチサイズ取得"""
        # training_settings を優先、フォールバックでmodel_settings
        training_batch = self.config.get("training_settings", {}).get("batch_size")
        if training_batch is not None:
            return training_batch
        
        return self.config.get("model_settings", {}).get("batch_size", 32)
    
    @property
    def learning_rate(self):
        """学習率取得"""
        # training_settings を優先、フォールバックでmodel_settings
        training_lr = self.config.get("training_settings", {}).get("learning_rate")
        if training_lr is not None:
            return training_lr
        
        return self.config.get("model_settings", {}).get("learning_rate", 0.001)
    
    @property
    def epochs(self):
        """エポック数取得"""
        # training_settings を優先、フォールバックでmodel_settings
        training_epochs = self.config.get("training_settings", {}).get("epochs")
        if training_epochs is not None:
            return training_epochs
        
        return self.config.get("model_settings", {}).get("epochs", 50)
    
    @property
    def use_core_only(self):
        """コア4クラスのみ使用フラグ"""
        return self.config.get("model_settings", {}).get("use_core_only", False)
    
    @property
    def use_12_classes(self):
        """12クラス使用フラグ"""
        return self.config.get("processing_settings", {}).get("use_12_classes", True)
    
    @property
    def memory_optimization(self):
        """メモリ最適化フラグ"""
        return self.config.get("training_settings", {}).get("memory_optimization", True)
    
    def get(self, key, default=None):
        """設定値取得"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def update_model_settings(self, **kwargs):
        """モデル設定更新"""
        if "model_settings" not in self.config:
            self.config["model_settings"] = {}
        self.config["model_settings"].update(kwargs)
        self._save_config()
    
    def _save_config(self):
        """設定ファイル保存"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"設定ファイル保存エラー: {e}")
    
    def validate_settings(self):
        """設定値の妥当性チェック（12クラス対応）"""
        errors = []
        
        # クラス数チェック
        class_order = self.class_order
        if len(class_order) != 12:
            errors.append(f"クラス数が不正です。期待値: 12, 実際: {len(class_order)}")
        
        # 必須クラス名チェック
        expected_classes = [
            "plastic", "plate", "byou", "concrete", "traverse", "kokudo",
            "gaiku_sankaku", "gaiku_setsu", "gaiku_takaku", "gaiku_hojo", 
            "traverse_in", "kagoshima_in"
        ]
        missing_classes = set(expected_classes) - set(class_order)
        if missing_classes:
            errors.append(f"必須クラスが不足: {list(missing_classes)}")
        
        # バッチサイズチェック
        if self.batch_size < 1 or self.batch_size > 512:
            errors.append(f"バッチサイズが不正です: {self.batch_size} (有効範囲: 1-512)")
        
        # 画像サイズチェック
        image_size = self.image_size
        if len(image_size) != 2 or image_size[0] != image_size[1]:
            errors.append(f"画像サイズが不正です: {image_size} (正方形である必要があります)")
        
        if image_size[0] < 32 or image_size[0] > 1024:
            errors.append(f"画像サイズが範囲外です: {image_size[0]} (有効範囲: 32-1024)")
        
        # 学習率チェック
        if self.learning_rate <= 0 or self.learning_rate > 1:
            errors.append(f"学習率が不正です: {self.learning_rate} (有効範囲: 0-1)")
        
        # バリデーション分割率チェック
        if self.validation_split < 0 or self.validation_split >= 1:
            errors.append(f"バリデーション分割率が不正です: {self.validation_split} (有効範囲: 0-1)")
        
        # メモリ使用率チェック
        if self.max_memory_usage < 0.1 or self.max_memory_usage > 1:
            errors.append(f"最大メモリ使用率が不正です: {self.max_memory_usage} (有効範囲: 0.1-1)")
        
        # モデルパスチェック
        if not self.model_path or not self.model_path.endswith(('.h5', '.keras')):
            errors.append(f"モデルパスが不正です: {self.model_path}")
        
        if errors:
            raise ValueError("設定検証エラー:\n" + "\n".join(f"- {error}" for error in errors))
        
        return True
    
    def get_trainer_config(self):
        """訓練用設定辞書取得"""
        return {
            'image_size': self.image_size,
            'batch_size': self.batch_size,
            'class_names': self.class_order,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'validation_split': self.validation_split,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'memory_optimization': self.memory_optimization
        }
    
    def get_app_config(self):
        """アプリケーション用設定辞書取得"""
        return {
            'model_path': self.model_path,
            'model_info_path': self.model_info_path,
            'class_names': self.class_order,
            'image_size': self.image_size,
            'batch_size': self.batch_size,
            'code_mapping': self.code_mapping,
            'num_classes': self.num_classes,
            'max_memory_usage': self.max_memory_usage
        }
    
    def print_summary(self):
        """設定サマリ表示"""
        print("\n=== ConfigLoader 設定サマリ ===")
        print(f"モデルタイプ: {self.get('app_info.model_type', 'Unknown')}")
        print(f"クラス数: {self.num_classes}")
        print(f"画像サイズ: {self.image_size}")
        print(f"バッチサイズ: {self.batch_size}")
        print(f"学習率: {self.learning_rate}")
        print(f"12クラス使用: {self.use_12_classes}")
        print(f"メモリ最適化: {self.memory_optimization}")
        print(f"設定ファイル: {self.config_path}")
        
        # 妥当性チェック結果
        try:
            self.validate_settings()
            print("✅ 設定検証: 正常")
        except ValueError as e:
            print(f"⚠️ 設定検証: エラー\n{e}")

# 使用例・統合テスト
if __name__ == "__main__":
    print("=== ConfigLoader 統合テスト ===")
    
    # 基本使用
    config = ConfigLoader()
    config.print_summary()
    
    # 訓練スクリプトでの使用例
    print("\n=== 訓練用設定 ===")
    trainer_config = config.get_trainer_config()
    for key, value in trainer_config.items():
        print(f"{key}: {value}")
    
    # アプリケーションでの使用例  
    print("\n=== アプリ用設定 ===")
    app_config = config.get_app_config()
    for key, value in app_config.items():
        print(f"{key}: {value}")
    
    # コードマッピング表示
    print("\n=== パイルコードマッピング ===")
    for class_name, code in config.code_mapping.items():
        print(f"{class_name} -> {code}")
    
    print("\n=== テスト完了 ===")
    print("これで訓練スクリプトとアプリケーション間の設定整合性が確保されます。")