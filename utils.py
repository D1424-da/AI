# ユーティリティ関数群
# Utility Functions

import json
import os
import shutil
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from PIL import Image

class ConfigManager:
    """設定管理クラス"""
    
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        """設定ファイル読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"設定ファイルが見つかりません: {self.config_path}")
            return self.get_default_config()
        except json.JSONDecodeError as e:
            print(f"設定ファイルの形式エラー: {e}")
            return self.get_default_config()
    
    def get_default_config(self):
        """デフォルト設定"""
        return {
            "pile_codes": {
                "plastic": "P",
                "plate": "PL", 
                "metal": "B",
                "concrete": "C",
                "target": "T",
                "kokudo": "KD",
                "gaiku_kijun": "GK",
                "gaiku_takaku": "GT",
                "gaiku_hojo": "GH",
                "gaiku_setsu": "GS"
            },
            "model_settings": {
                "confidence_threshold": 0.7,
                "image_size": [224, 224]
            },
            "processing_settings": {
                "supported_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
                "backup_enabled": True
            }
        }
    
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

class FileManager:
    """ファイル管理クラス"""
    
    @staticmethod
    def create_backup(source_folder, backup_suffix="_backup"):
        """フォルダのバックアップ作成"""
        try:
            source_path = Path(source_folder)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{source_path.name}{backup_suffix}_{timestamp}"
            backup_path = source_path.parent / backup_name
            
            shutil.copytree(source_path, backup_path)
            return str(backup_path)
        except Exception as e:
            print(f"バックアップ作成エラー: {e}")
            return None
    
    @staticmethod
    def get_image_files(folder_path, extensions=None):
        """フォルダ内の画像ファイル一覧取得"""
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        folder = Path(folder_path)
        image_files = []
        
        for ext in extensions:
            # 小文字と大文字両方をチェック
            pattern_lower = f"**/*{ext.lower()}"
            pattern_upper = f"**/*{ext.upper()}"
            image_files.extend(folder.glob(pattern_lower))
            image_files.extend(folder.glob(pattern_upper))
        
        # 重複を除去
        return list(set(image_files))
    
    @staticmethod
    def is_valid_image(image_path):
        """画像ファイルの有効性チェック"""
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    @staticmethod
    def generate_new_filename(original_path, pile_code):
        """新しいファイル名生成"""
        path_obj = Path(original_path)
        original_name = path_obj.stem
        extension = path_obj.suffix
        
        # 既存のコードをチェック・置換
        existing_codes = ['P', 'PL', 'B', 'C', 'T', 'KD', 'GK', 'GT', 'GH', 'GS']
        
        for code in existing_codes:
            if original_name.startswith(f"{code}"):
                # 既存コードを新しいコードに置換
                new_name = f"{pile_code}{original_name[len(code):]}"
                return path_obj.parent / f"{new_name}{extension}"
        
        # 新規にコード追加
        new_name = f"{pile_code}{original_name}"
        return path_obj.parent / f"{new_name}{extension}"

class ImageProcessor:
    """画像処理クラス"""
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def preprocess_image(self, image_path):
        """画像前処理（モデル入力用）"""
        try:
            # 画像読み込み
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # BGRからRGBに変換
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # リサイズ
            image = cv2.resize(image, self.target_size)
            
            # 正規化
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            print(f"画像前処理エラー: {image_path} - {e}")
            return None
    
    def preprocess_batch(self, image_paths):
        """バッチ画像前処理"""
        batch_images = []
        valid_paths = []
        
        for image_path in image_paths:
            processed = self.preprocess_image(image_path)
            if processed is not None:
                batch_images.append(processed)
                valid_paths.append(image_path)
        
        if batch_images:
            return np.array(batch_images), valid_paths
        else:
            return None, []

class Logger:
    """ログ管理クラス"""
    
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self.logs = []
    
    def log(self, message, level="INFO"):
        """ログメッセージ記録"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        self.logs.append(log_entry)
        
        # コールバック実行（GUI更新用）
        if self.log_callback:
            self.log_callback(log_entry)
        
        # コンソール出力
        print(log_entry)
    
    def info(self, message):
        """情報ログ"""
        self.log(message, "INFO")
    
    def warning(self, message):
        """警告ログ"""
        self.log(message, "WARNING")
    
    def error(self, message):
        """エラーログ"""
        self.log(message, "ERROR")
    
    def save_logs(self, filepath):
        """ログファイル保存"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for log_entry in self.logs:
                    f.write(log_entry + '\n')
        except Exception as e:
            self.error(f"ログ保存エラー: {e}")

class PerformanceMonitor:
    """パフォーマンス監視クラス"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.processing_times = []
    
    def start(self):
        """処理時間計測開始"""
        self.start_time = datetime.now()
    
    def end(self):
        """処理時間計測終了"""
        if self.start_time:
            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds()
            self.processing_times.append(duration)
            return duration
        return 0
    
    def get_stats(self):
        """統計情報取得"""
        if not self.processing_times:
            return None
        
        return {
            'total_time': sum(self.processing_times),
            'average_time': np.mean(self.processing_times),
            'min_time': min(self.processing_times),
            'max_time': max(self.processing_times),
            'count': len(self.processing_times)
        }

# ユーティリティ関数
def validate_model_path(model_path):
    """モデルファイルの存在確認"""
    return Path(model_path).exists()

def get_model_info(model_path):
    """モデル情報取得"""
    info_path = Path(model_path).parent / "all_pile_model_info.json"
    if info_path.exists():
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return None

def format_file_size(size_bytes):
    """ファイルサイズを人間が読みやすい形式に変換"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def check_disk_space(path, required_mb=100):
    """ディスク容量チェック"""
    try:
        stat = shutil.disk_usage(path)
        free_mb = stat.free / (1024 * 1024)
        return free_mb >= required_mb
    except:
        return True  # チェックできない場合は通す
