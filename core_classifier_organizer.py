"""
コアクラス画像分類・整理アプリ
訓練済みモデルを使って画像を分類し、フォルダ分け・リネームを行います
"""

import os
import sys
import json
import shutil
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from PIL import Image, ImageTk
import cv2
import numpy as np

# TensorFlowの警告を抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=== TensorFlow Import Debug ===")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current directory: {os.getcwd()}")
print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

# スクリプトのあるディレクトリに移動
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != script_dir:
    print(f"Changing directory from {os.getcwd()} to {script_dir}")
    os.chdir(script_dir)
    print(f"New current directory: {os.getcwd()}")

print("Starting TensorFlow import...")

# 仮想環境確認
venv_path = os.environ.get('VIRTUAL_ENV')
if venv_path:
    print(f"Virtual environment: {venv_path}")
else:
    print("No virtual environment detected")

# Pythonパス確認
print(f"Python path: {sys.path[:3]}...")  # 最初の3つのパスを表示

try:
    print("Step 1: Importing tensorflow...")
    import tensorflow as tf
    print(f"Step 2: TensorFlow version: {tf.__version__}")
    
    print("Step 3: Importing keras...")
    from tensorflow import keras
    print(f"Step 4: Keras version: {keras.__version__}")
    
    print("Step 5: Setting TF_AVAILABLE = True")
    TF_AVAILABLE = True
    print(f"SUCCESS: TensorFlow loaded successfully: version {tf.__version__}")
    
except ImportError as e:
    print(f"ImportError: {str(e)}")
    print("Please install TensorFlow with: pip install tensorflow")
    TF_AVAILABLE = False
except Exception as e:
    print(f"Exception: {str(e)}")
    import traceback
    traceback.print_exc()
    TF_AVAILABLE = False

print(f"Final TF_AVAILABLE status: {TF_AVAILABLE}")
print("=== TensorFlow Import Complete ===\n")

class CoreClassifierOrganizer:
    """コアクラス分類・整理アプリケーション"""
    
    def __init__(self, root):
        # TensorFlow利用可能性をクラス変数として保存
        self.tf_available = TF_AVAILABLE
        self.root = root
        self.root.title("コアクラス画像分類・整理アプリ v1.0")
        self.root.geometry("900x700")
        
        # コアクラス定義（4クラス）
        self.core_classes = [
            'plastic',      # プラスチック杭
            'plate',        # プレート
            'byou',         # 金属鋲
            'concrete',     # コンクリート杭
        ]
        
        # 拡張クラス定義（12クラス）
        self.extended_classes = [
            'plastic', 'plate', 'byou', 'concrete',
            'traverse', 'kokudo', 'gaiku_sankaku', 'gaiku_setsu',
            'gaiku_takaku', 'gaiku_hojo', 'traverse_in', 'kagoshima_in'
        ]
        
        # 杭種コード定義（12クラス対応）
        self.pile_codes = {
            'plastic': 'P', 'plate': 'PL', 'byou': 'B', 'concrete': 'C',
            'traverse': 'T', 'kokudo': 'KD', 'gaiku_sankaku': 'GS',
            'gaiku_setsu': 'GT', 'gaiku_takaku': 'GH', 'gaiku_hojo': 'GK',
            'traverse_in': 'TI', 'kagoshima_in': 'KI'
        }
        
        # AIモデル（複数対応）
        self.model = None
        self.core_model = None
        self.extended_model = None
        self.model_info = None
        self.class_names = self.core_classes  # デフォルトはコア4クラス
        self.use_core_only = True
        
        # 処理統計（動的クラス対応）
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'classified': {},  # 動的に設定
            'moved_files': 0,
            'renamed_files': 0,
            'error_files': 0,
            'low_confidence_skipped': 0,  # 低信頼度スキップファイル数
            'skipped_files': []  # スキップされたファイル詳細
        }
        
        # 設定
        self.confidence_threshold = 0.1
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        """UIセットアップ"""
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # タイトル
        title_label = ttk.Label(main_frame, text="コアクラス画像分類・整理アプリ", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 入力フォルダ選択
        input_frame = ttk.LabelFrame(main_frame, text="入力設定", padding="10")
        input_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(input_frame, text="分類対象フォルダ:").grid(row=0, column=0, sticky=tk.W)
        self.input_folder_var = tk.StringVar()
        input_entry = ttk.Entry(input_frame, textvariable=self.input_folder_var, width=60)
        input_entry.grid(row=0, column=1, padx=(10, 0))
        ttk.Button(input_frame, text="選択", command=self.select_input_folder).grid(row=0, column=2, padx=(10, 0))
        
        # 出力フォルダ選択
        ttk.Label(input_frame, text="整理先フォルダ:").grid(row=1, column=0, sticky=tk.W)
        self.output_folder_var = tk.StringVar()
        output_entry = ttk.Entry(input_frame, textvariable=self.output_folder_var, width=60)
        output_entry.grid(row=1, column=1, padx=(10, 0))
        ttk.Button(input_frame, text="選択", command=self.select_output_folder).grid(row=1, column=2, padx=(10, 0))
        
        # 処理オプション
        options_frame = ttk.LabelFrame(main_frame, text="処理オプション", padding="10")
        options_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 信頼度閾値
        ttk.Label(options_frame, text="信頼度閾値:").grid(row=0, column=0, sticky=tk.W)
        self.confidence_var = tk.DoubleVar(value=0.1)
        confidence_scale = ttk.Scale(options_frame, from_=0.1, to=1.0, 
                                   variable=self.confidence_var, orient=tk.HORIZONTAL, length=200)
        confidence_scale.grid(row=0, column=1, padx=(10, 0))
        self.confidence_label = ttk.Label(options_frame, text="0.10")
        self.confidence_label.grid(row=0, column=2, padx=(10, 0))
        confidence_scale.configure(command=self.update_confidence_label)
        
        # その他オプション
        self.create_backup_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="処理前にバックアップを作成", 
                       variable=self.create_backup_var).grid(row=1, column=0, columnspan=2, sticky=tk.W)
        
        self.add_prefix_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="ファイル名に杭種コードを追加", 
                       variable=self.add_prefix_var).grid(row=2, column=0, columnspan=2, sticky=tk.W)
        
        self.copy_files_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="ファイルをコピー（移動ではなく）", 
                       variable=self.copy_files_var).grid(row=3, column=0, columnspan=2, sticky=tk.W)
        
        self.create_lowconf_folder_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="低信頼度ファイル用フォルダを作成", 
                       variable=self.create_lowconf_folder_var).grid(row=4, column=0, columnspan=2, sticky=tk.W)
        
        # モデル情報表示
        model_frame = ttk.LabelFrame(main_frame, text="モデル情報", padding="10")
        model_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.model_info_label = ttk.Label(model_frame, text="モデル: 読み込み中...")
        self.model_info_label.grid(row=0, column=0, sticky=tk.W)
        
        # 実行ボタン
        execute_frame = ttk.Frame(main_frame)
        execute_frame.grid(row=4, column=0, columnspan=3, pady=20)
        
        self.execute_button = ttk.Button(execute_frame, text="分類・整理実行", 
                                        command=self.execute_classification, 
                                        style="Accent.TButton")
        self.execute_button.grid(row=0, column=0, padx=(0, 10))
        
        ttk.Button(execute_frame, text="プレビュー", 
                  command=self.preview_classification).grid(row=0, column=1, padx=(0, 10))
        
        ttk.Button(execute_frame, text="統計リセット", 
                  command=self.reset_stats).grid(row=0, column=2)
        
        # プログレスバー
        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, 
                                      maximum=100, length=500)
        progress_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # ステータス表示
        self.status_var = tk.StringVar(value="準備完了")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=6, column=0, columnspan=3)
        
        # 統計表示
        stats_frame = ttk.LabelFrame(main_frame, text="分類統計", padding="10")
        stats_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.stats_text = tk.Text(stats_frame, height=6, width=80)
        stats_scrollbar = ttk.Scrollbar(stats_frame, orient="vertical", command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        stats_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # ログエリア
        log_frame = ttk.LabelFrame(main_frame, text="処理ログ", padding="10")
        log_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        self.log_text = tk.Text(log_frame, height=12, width=80)
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # グリッドの重み設定
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(8, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.rowconfigure(0, weight=1)
    
    def update_confidence_label(self, value):
        """信頼度ラベル更新"""
        self.confidence_label.config(text=f"{float(value):.2f}")
        self.confidence_threshold = float(value)
    
    def log_message(self, message):
        """ログメッセージを表示"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"
            self.log_text.insert(tk.END, log_entry)
            self.log_text.see(tk.END)
            self.root.update_idletasks()
        except Exception as e:
            print(f"Logging error: {e}")
    
    def update_stats_display(self):
        """統計表示を更新（動的クラス対応）"""
        stats_text = "=== 分類統計 ===\n"
        stats_text += f"総ファイル数: {self.stats['total_files']}\n"
        stats_text += f"処理済み: {self.stats['processed_files']}\n"
        stats_text += f"移動済み: {self.stats['moved_files']}\n"
        stats_text += f"リネーム済み: {self.stats['renamed_files']}\n"
        stats_text += f"エラー: {self.stats['error_files']}\n"
        stats_text += f"低信頼度スキップ: {self.stats['low_confidence_skipped']}\n\n"
        
        # 使用中のモデル情報
        model_type = "4クラス" if self.use_core_only else f"{len(self.class_names)}クラス"
        stats_text += f"モデルタイプ: {model_type}\n"
        
        stats_text += "クラス別分類結果:\n"
        for class_name, count in self.stats['classified'].items():
            percentage = (count / max(1, self.stats['processed_files'])) * 100
            pile_code = self.pile_codes.get(class_name, '?')
            stats_text += f"  {class_name}({pile_code}): {count}枚 ({percentage:.1f}%)\n"
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
    
    def select_input_folder(self):
        """入力フォルダ選択"""
        folder = filedialog.askdirectory(title="分類対象フォルダを選択してください")
        if folder:
            self.input_folder_var.set(folder)
            self.log_message(f"入力フォルダ: {folder}")
    
    def select_output_folder(self):
        """出力フォルダ選択"""
        folder = filedialog.askdirectory(title="整理先フォルダを選択してください")
        if folder:
            self.output_folder_var.set(folder)
            self.log_message(f"出力フォルダ: {folder}")
    
    def load_model(self):
        """AIモデル読み込み（4クラス・12クラス両対応）"""
        try:
            self.log_message("=== モデル読み込み開始（段階的訓練対応）===")
            self.log_message(f"TensorFlow利用可能: {self.tf_available}")
            
            if self.tf_available:
                try:
                    import tensorflow as tf
                    self.log_message(f"TensorFlow: {tf.__version__}")
                except Exception as tf_error:
                    self.log_message(f"TensorFlow確認エラー: {tf_error}")
                    self.tf_available = False
            
            if not self.tf_available:
                self.log_message("⚠️  TensorFlow未対応 - デモモードで動作")
                self._setup_demo_mode()
                return
            
            # 優先順位でモデル検索（段階的訓練対応）
            model_candidates = [
                # 1. 最新の12クラスモデル（最優先）
                ("models/all_pile_classifier.h5", "extended"),
                ("all_pile_classifier.h5", "extended"),
                
                # 2. 拡張クラス対応モデル
                ("models/pile_classifier_extended.h5", "extended"),
                ("models/extended_classifier.h5", "extended"),
                
                # 3. コア4クラス専用モデル
                ("models/pile_classifier.h5", "core"),
                ("models/core_classifier.h5", "core"),
                ("models/balanced_core_classifier.h5", "core"),
                ("models/realistic_core_classifier.h5", "core"),
                ("pile_classifier.h5", "core"),
                ("core_classifier.h5", "core")
            ]
            
            # 存在チェックログ
            self.log_message("📁 モデルファイル存在チェック:")
            for model_path, model_type in model_candidates:
                exists = os.path.exists(model_path)
                status = "✅" if exists else "❌"
                self.log_message(f"  {status} {model_path} ({model_type})")
            
            loaded = False
            for model_path, expected_type in model_candidates:
                if os.path.exists(model_path):
                    if self._try_load_model(model_path, expected_type):
                        loaded = True
                        break
            
            # フォールバック処理
            if not loaded:
                self.log_message("⚠️  訓練済みモデルが見つかりません")
                self._setup_demo_mode()
                messagebox.showwarning("警告", 
                    "訓練済みモデルが見つかりません。\n"
                    "先にモデルを訓練するか、デモモードで実行してください。")
            
        except Exception as e:
            self.log_message(f"❌ モデル読み込みエラー: {str(e)}")
            messagebox.showerror("エラー", f"モデル読み込み失敗:\n{str(e)}")
    
    def _try_load_model(self, model_path, expected_type):
        """個別モデル読み込み試行"""
        try:
            self.log_message(f"🔄 モデル読み込み試行: {model_path} ({expected_type})")
            
            from tensorflow import keras
            model = keras.models.load_model(model_path)
            
            # モデル情報取得
            model_class_names = self._load_model_info(model_path, expected_type)
            output_classes = model.output_shape[-1]
            
            # モデルタイプ判定・設定
            actual_type = self._determine_model_type(output_classes, model_class_names, expected_type)
            
            if actual_type == "core":
                self._setup_core_model(model, model_class_names, model_path)
            elif actual_type == "extended":
                self._setup_extended_model(model, model_class_names, model_path)
            else:
                self.log_message(f"❌ 未対応のモデルタイプ: {actual_type}")
                return False
            
            # 統計辞書を現在のクラスに合わせて初期化
            self.stats['classified'] = {class_name: 0 for class_name in self.class_names}
            
            self.log_message(f"✅ モデル読み込み成功: {Path(model_path).stem}")
            self.log_message(f"   📊 出力クラス数: {output_classes}")
            self.log_message(f"   🎯 使用クラス: {self.class_names}")
            self.log_message(f"   ⚙️  総パラメータ: {model.count_params():,}")
            
            return True
            
        except Exception as e:
            self.log_message(f"❌ {Path(model_path).name} 読み込み失敗: {str(e)}")
            return False
    
    def _setup_demo_mode(self):
        """デモモード設定"""
        self.model = None
        self.use_core_only = True
        self.class_names = self.core_classes
        self.stats['classified'] = {class_name: 0 for class_name in self.class_names}
        self.model_info_label.config(text="モデル: デモモード（ランダム予測）")
    
    def _setup_core_model(self, model, model_class_names, model_path):
        """コア4クラスモデル設定"""
        self.core_model = model
        self.model = model
        self.use_core_only = True
        self.class_names = model_class_names if model_class_names else self.core_classes
        
        model_name = Path(model_path).stem
        self.model_info_label.config(text=f"モデル: {model_name} | コア4クラス")
    
    def _setup_extended_model(self, model, model_class_names, model_path):
        """拡張12クラスモデル設定"""
        self.extended_model = model
        self.model = model
        self.use_core_only = False
        self.class_names = model_class_names if model_class_names else self.extended_classes
        
        model_name = Path(model_path).stem
        self.model_info_label.config(text=f"モデル: {model_name} | 拡張{len(self.class_names)}クラス")
    
    def _determine_model_type(self, output_classes, model_class_names, expected_type):
        """モデルタイプ判定"""
        # クラス数による判定
        if output_classes == 4:
            return "core"
        elif output_classes == 12:
            return "extended"
        
        # クラス名リストによる判定
        if model_class_names:
            if len(model_class_names) == 4:
                return "core"
            elif len(model_class_names) == 12:
                return "extended"
        
        # 期待値による判定
        return expected_type
    
    def _load_model_info(self, model_path, model_type):
        """モデル情報読み込み（正しいクラス順序取得）"""
        try:
            # 情報ファイル優先順位
            if model_type == "core":
                info_files = [
                    Path(model_path).parent / "core_model_info.json",
                    Path(model_path).parent / "model_info.json"
                ]
            else:  # extended
                info_files = [
                    Path(model_path).parent / "all_pile_model_info.json",
                    Path(model_path).parent / "model_info.json"
                ]
            
            for info_path in info_files:
                if info_path.exists():
                    try:
                        with open(info_path, 'r', encoding='utf-8') as f:
                            model_info = json.load(f)
                        
                        # 正しいクラス順序取得
                        if 'label_encoder_classes' in model_info:
                            class_names = model_info['label_encoder_classes']
                            self.log_message(f"   📋 正確なクラス順序取得: {class_names}")
                            self.log_message(f"   📄 情報源: {info_path.name}")
                            
                            # 画像サイズ設定
                            if 'image_size' in model_info:
                                if isinstance(model_info['image_size'], list):
                                    self.image_size = tuple(model_info['image_size'])
                                else:
                                    self.image_size = model_info['image_size']
                                self.log_message(f"   📐 画像サイズ: {self.image_size}")
                            
                            return class_names
                            
                    except Exception as e:
                        self.log_message(f"   ⚠️  情報ファイル読み込み警告: {info_path.name} - {str(e)}")
            
            self.log_message(f"   🚨 モデル情報未発見 - デフォルト順序使用")
            return None
            
        except Exception as e:
            self.log_message(f"❌ モデル情報読み込みエラー: {str(e)}")
            return None
    
    def load_model_info(self, model_path):
        """モデル情報読み込み（簡素化版）"""
        try:
            # 既に_load_model_infoで処理済みのため、ここでは簡単なログのみ
            self.log_message(f"📋 モデル情報読み込み完了: {Path(model_path).name}")
            
            # デフォルト設定（未設定の場合）
            if not hasattr(self, 'image_size') or self.image_size is None:
                self.image_size = (320, 320)  # 12クラスモデル標準
                self.log_message(f"� デフォルト画像サイズ設定: {self.image_size}")
                
        except Exception as e:
            self.log_message(f"❌ モデル情報処理エラー: {str(e)}")
            self.image_size = (320, 320)
    
    def preprocess_image(self, image_path):
        """画像前処理（日本語パス対応・決定論的処理）"""
        try:
            # ファイルパスを絶対パスに変換
            abs_path = Path(image_path).resolve()
            self.log_message(f"📁 画像読み込み開始: {abs_path.name}")
            self.log_message(f"   📍 読み込み対象: {abs_path}")
            
            # ファイル存在確認
            if not abs_path.exists():
                self.log_message(f"   ❌ ファイルが存在しません: {abs_path}")
                return None
            
            # 日本語パス対応: PIL経由で画像読み込み
            try:
                from PIL import Image as PILImage
                pil_image = PILImage.open(abs_path)
                
                # PIL → OpenCV形式に変換
                if pil_image.mode == 'RGB':
                    image = np.array(pil_image)  # RGB
                elif pil_image.mode == 'RGBA':
                    image = np.array(pil_image)  # RGBA
                elif pil_image.mode == 'L':
                    image = np.array(pil_image)  # Grayscale
                else:
                    # その他のモードはRGBに変換
                    pil_image = pil_image.convert('RGB')
                    image = np.array(pil_image)
                
                self.log_message(f"   ✅ PIL読み込み成功: {pil_image.mode}, {pil_image.size}")
                self.log_message(f"   📊 読み込み後: shape={image.shape}, dtype={image.dtype}")
                
            except Exception as e:
                self.log_message(f"   ❌ PIL読み込み失敗: {abs_path} - {e}")
                
                # フォールバック: OpenCV直接読み込み
                image = cv2.imread(str(abs_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    self.log_message(f"   ❌ OpenCV読み込みも失敗: {abs_path}")
                    return None
                self.log_message(f"   ⚠️  OpenCVフォールバック成功: shape={image.shape}")

            # PIL読み込み済みの場合はRGB形式、OpenCV読み込みの場合はBGR形式
            if len(image.shape) == 2:
                # モノクロ画像 → RGB変換
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                self.log_message(f"   🔄 Grayscale→RGB変換")
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # RGBA → RGB変換（PIL読み込み）
                image = image[:, :, :3]  # アルファチャンネル除去
                self.log_message(f"   🔄 RGBA→RGB変換")
            elif len(image.shape) == 3 and image.shape[2] == 3:
                # 既にRGB形式（PIL）またはBGR形式（OpenCV）
                # PILの場合は既にRGB、OpenCVの場合のみBGR→RGB変換が必要
                # しかし、PIL経由なので変換不要
                self.log_message(f"   ✅ RGB形式確認: {image.shape}")
            else:
                self.log_message(f"   ⚠️  未対応画像形状: {image.shape}")
                return None
            
            # リサイズ（補間方法を明示的に指定して一貫性確保）
            self.log_message(f"   🔄 リサイズ: {image.shape[:2]} → {self.image_size}")
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
            
            # データ型を確実にfloat32に変換
            image = image.astype(np.float32)
            
            # 正規化（0-1範囲）
            image = image / 255.0
            
            # 値の範囲チェック（デバッグ用）
            if np.any(image < 0) or np.any(image > 1):
                self.log_message(f"   ⚠️  正規化後の値が範囲外: min={np.min(image):.6f}, max={np.max(image):.6f}")
            
            # バッチ次元追加
            image = np.expand_dims(image, axis=0)
            
            # 最終形状確認
            expected_shape = (1, self.image_size[1], self.image_size[0], 3)
            if image.shape != expected_shape:
                self.log_message(f"   ❌ 前処理後形状エラー: 期待{expected_shape}, 実際{image.shape}")
                return None
            
            self.log_message(f"   ✅ 前処理完了: final_shape={image.shape}")
            return image
            
        except Exception as e:
            self.log_message(f"   ❌ 画像前処理エラー: {Path(image_path).name} - {str(e)}")
            import traceback
            self.log_message(f"   📜 前処理エラー詳細: {traceback.format_exc()}")
            return None
    
    def classify_image(self, image_path):
        """画像分類（完全決定論的・安定化）"""
        try:
            self.log_message(f"\n🔍 **分類開始**: {Path(image_path).name}")
            
            if self.model is None:
                # デモモード（ランダム予測）
                self.log_message("⚠️  デモモード: ランダム予測を使用")
                import random
                predicted_class = random.choice(self.class_names)
                confidence = random.uniform(0.6, 0.9)
                self.log_message(f"   🎲 デモ結果: {predicted_class} (信頼度: {confidence:.3f})")
                return predicted_class, confidence
            
            # 前処理
            processed_image = self.preprocess_image(image_path)
            if processed_image is None:
                self.log_message(f"   ❌ 前処理失敗: {Path(image_path).name}")
                return None, 0.0
            
            # **推論安定化**: 複数の安定化手法を適用
            
            # 1. TensorFlowの決定論的実行
            try:
                if self.tf_available:
                    import tensorflow as tf
                    tf.config.experimental.enable_op_determinism()
            except:
                pass  # 既に設定済みの場合
            
            # 2. モデルを評価モードに設定（ドロップアウト完全無効化）
            if hasattr(self.model, 'training'):
                self.model.training = False
            
            # 3. 予測実行（安定化された環境）
            if self.tf_available:
                import tensorflow as tf
                with tf.device('/CPU:0'):  # CPU使用で最大安定化
                    # **重要**: training=Falseを明示的に指定
                    predictions = self.model(processed_image, training=False)
                    # 予測結果を即座にnumpy配列に変換（メモリ安定化）
                    pred_array = predictions.numpy()[0]
            else:
                # TensorFlowが利用できない場合のフォールバック
                predictions = self.model.predict(processed_image, verbose=0)
                pred_array = predictions[0]
            
            # 予測結果の詳細ログ（再現性確認用）
            self.log_message(f"   🔍 Raw predictions: {[f'{p:.6f}' for p in pred_array]}")
            self.log_message(f"   📊 Prediction sum: {np.sum(pred_array):.6f}")
            
            # 最も高い確率のクラスを取得
            predicted_class_idx = np.argmax(pred_array)
            confidence = float(pred_array[predicted_class_idx])
            
            # クラス名取得
            if predicted_class_idx < len(self.class_names):
                predicted_class = self.class_names[predicted_class_idx]
            else:
                self.log_message(f"   ❌ クラスインデックス範囲外: {predicted_class_idx}")
                return None, 0.0
            
            # 全クラスの確率を表示（デバッグ用）
            self.log_message("   📈 全クラス確率:")
            for i, class_name in enumerate(self.class_names):
                prob = pred_array[i]
                pile_code = self.pile_codes.get(class_name, '?')
                marker = " ✅" if i == predicted_class_idx else ""
                self.log_message(f"      {class_name}({pile_code}): {prob:.4f}{marker}")
            
            self.log_message(f"\n   ✅ **最終結果**: {predicted_class} (信頼度: {confidence:.4f})")
            
            # 信頼度チェック
            if confidence < 0.3:
                self.log_message(f"   ⚠️  低信頼度警告: {confidence:.4f}")
            elif confidence < 0.7:
                self.log_message(f"   ⚡ 中信頼度注意: {confidence:.4f}")
            
            return predicted_class, confidence
            
        except Exception as e:
            import traceback
            error_msg = f"分類処理エラー: {str(e)}"
            self.log_message(f"   ❌ {error_msg}")
            self.log_message(f"   📜 Traceback: {traceback.format_exc()}")
            return None, 0.0
    
    def generate_new_filename(self, original_path, predicted_class):
        """新しいファイル名生成（12クラス対応）"""
        try:
            path_obj = Path(original_path)
            original_name = path_obj.stem
            extension = path_obj.suffix
            
            if self.add_prefix_var.get():
                pile_code = self.pile_codes.get(predicted_class, 'U')
                
                # 全杭種コードに対応（長いコードを優先チェック）
                existing_codes = list(self.pile_codes.values())
                existing_codes.sort(key=len, reverse=True)  # 長さ順でソート
                
                for code in existing_codes:
                    if original_name.startswith(code):
                        new_name = f"{pile_code}{original_name[len(code):]}" 
                        return f"{new_name}{extension}"
                
                # 新規コード追加
                new_name = f"{pile_code}{original_name}"
                return f"{new_name}{extension}"
            else:
                # コード追加なし
                return path_obj.name
                
        except Exception as e:
            self.log_message(f"ファイル名生成エラー: {str(e)}")
            return Path(original_path).name
    
    def _handle_low_confidence_file(self, image_file, predicted_class, confidence, output_folder):
        """低信頼度ファイルの処理"""
        try:
            if not self.create_lowconf_folder_var.get():
                # 低信頼度フォルダ作成が無効の場合は単純スキップ
                self.log_message(f"   📤 スキップ: 低信頼度フォルダ作成無効")
                return True
            
            # 低信頼度フォルダに移動/コピー
            lowconf_folder = Path(output_folder) / "low_confidence"
            
            # ファイル名生成（杭種コードチェックボックスに従う）
            original_name = image_file.stem
            extension = image_file.suffix
            confidence_str = f"{confidence:.3f}".replace(".", "_")
            
            # 杭種コード追加のチェックボックス設定に従う
            if self.add_prefix_var.get():
                pile_code = self.pile_codes.get(predicted_class, 'U')
                new_filename = f"{pile_code}_LOWCONF_{predicted_class}_{confidence_str}_{original_name}{extension}"
            else:
                # 杭種コード追加が無効の場合は、LOWCONプレフィックスも付けない
                new_filename = f"{original_name}{extension}"
            
            dest_path = lowconf_folder / new_filename
            
            # 重複ファイル名対応
            counter = 1
            original_dest_path = dest_path
            while dest_path.exists():
                stem = original_dest_path.stem
                suffix = original_dest_path.suffix
                dest_path = original_dest_path.parent / f"{stem}_{counter}{suffix}"
                counter += 1
            
            # ファイル移動またはコピー
            if self.copy_files_var.get():
                shutil.copy2(image_file, dest_path)
                action = "コピー"
            else:
                shutil.move(str(image_file), dest_path)
                action = "移動"
            
            self.log_message(f"   📋 {action}: {image_file.name} → low_confidence/{dest_path.name}")
            return True
            
        except Exception as e:
            self.log_message(f"   ❌ 低信頼度ファイル処理エラー: {str(e)}")
            return False
    
    def create_backup(self, source_folder):
        """バックアップ作成"""
        try:
            if not self.create_backup_var.get():
                return True
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_folder = Path(source_folder).parent / f"backup_classify_{timestamp}"
            shutil.copytree(source_folder, backup_folder)
            self.log_message(f"バックアップ作成: {backup_folder}")
            return True
            
        except Exception as e:
            self.log_message(f"バックアップ作成エラー: {str(e)}")
            return False
    
    def create_class_folders(self, output_folder):
        """クラス別フォルダ作成（動的クラス対応）"""
        try:
            output_path = Path(output_folder)
            
            # メインクラスフォルダ作成
            for class_name in self.class_names:
                class_folder = output_path / class_name
                class_folder.mkdir(parents=True, exist_ok=True)
                self.log_message(f"クラスフォルダ作成: {class_folder}")
            
            # 低信頼度ファイル用フォルダ作成
            if self.create_lowconf_folder_var.get():
                lowconf_folder = output_path / "low_confidence"
                lowconf_folder.mkdir(parents=True, exist_ok=True)
                self.log_message(f"低信頼度フォルダ作成: {lowconf_folder}")
            
            self.log_message(f"✅ {len(self.class_names)}クラス分のフォルダを作成完了")
            return True
            
        except Exception as e:
            self.log_message(f"フォルダ作成エラー: {str(e)}")
            return False
    
    def preview_classification(self):
        """分類プレビュー"""
        input_folder = self.input_folder_var.get()
        if not input_folder or not os.path.exists(input_folder):
            messagebox.showwarning("警告", "入力フォルダを選択してください")
            return
        
        # 最初の5枚をプレビュー
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            pattern = f"**/*{ext}"
            image_files.extend(list(Path(input_folder).glob(pattern)))
            if len(image_files) >= 5:
                break
        
        if not image_files:
            messagebox.showinfo("情報", "処理対象の画像ファイルが見つかりません")
            return
        
        # プレビューウィンドウ
        preview_window = tk.Toplevel(self.root)
        preview_window.title("分類プレビュー")
        preview_window.geometry("600x400")
        
        preview_text = tk.Text(preview_window, wrap=tk.WORD)
        preview_scrollbar = ttk.Scrollbar(preview_window, orient="vertical", command=preview_text.yview)
        preview_text.configure(yscrollcommand=preview_scrollbar.set)
        
        preview_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        preview_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        preview_text.insert(tk.END, "=== 分類プレビュー（最初の5枚） ===\n\n")
        
        for i, image_file in enumerate(image_files[:5]):
            predicted_class, confidence = self.classify_image(image_file)
            new_filename = self.generate_new_filename(image_file, predicted_class)
            
            preview_text.insert(tk.END, f"{i+1}. {image_file.name}\n")
            preview_text.insert(tk.END, f"   予測: {predicted_class} (信頼度: {confidence:.3f})\n")
            preview_text.insert(tk.END, f"   新ファイル名: {new_filename}\n")
            preview_text.insert(tk.END, f"   移動先: {predicted_class}/\n\n")
    
    def reset_stats(self):
        """統計リセット（動的クラス対応）"""
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'classified': {class_name: 0 for class_name in self.class_names},
            'moved_files': 0,
            'renamed_files': 0,
            'error_files': 0,
            'low_confidence_skipped': 0,
            'skipped_files': []
        }
        self.update_stats_display()
        model_info = f"{len(self.class_names)}クラス" if self.class_names else "Unknown"
        self.log_message(f"統計をリセットしました ({model_info})")
    
    def execute_classification(self):
        """分類・整理実行"""
        input_folder = self.input_folder_var.get()
        output_folder = self.output_folder_var.get()
        
        if not input_folder or not output_folder:
            messagebox.showwarning("警告", "入力フォルダと出力フォルダを選択してください")
            return
        
        if not os.path.exists(input_folder):
            messagebox.showerror("エラー", "入力フォルダが存在しません")
            return
        
        # 別スレッドで処理実行
        self.execute_button.config(state="disabled")
        thread = threading.Thread(target=self._process_classification, args=(input_folder, output_folder))
        thread.daemon = True
        thread.start()
    
    def _process_classification(self, input_folder, output_folder):
        """分類処理メインロジック"""
        try:
            self.status_var.set("処理開始...")
            self.log_message("=== 分類・整理処理開始 ===")
            
            # バックアップ作成
            if not self.create_backup(input_folder):
                if not messagebox.askyesno("確認", "バックアップの作成に失敗しました。処理を続行しますか？"):
                    return
            
            # 出力フォルダとクラス別フォルダ作成
            if not self.create_class_folders(output_folder):
                messagebox.showerror("エラー", "出力フォルダの作成に失敗しました")
                return
            
            # 画像ファイル収集（重複排除対応）
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_files_set = set()  # 重複排除用
            
            self.log_message("画像ファイル検索開始")
            self.log_message(f"検索対象フォルダ: {input_folder}")
            self.log_message(f"検索対象拡張子: {image_extensions}")
            
            # 拡張子別にファイル収集
            extension_counts = {}
            for ext in image_extensions:
                for case_ext in [ext, ext.upper()]:
                    pattern = f"**/*{case_ext}"
                    found_files = list(Path(input_folder).glob(pattern))
                    
                    if found_files:
                        extension_counts[case_ext] = len(found_files)
                        # setに追加して重複排除
                        for file_path in found_files:
                            image_files_set.add(file_path.resolve())  # 絶対パスで統一
            
            # 結果をリストに変換
            image_files = list(image_files_set)
            
            # 詳細ログ出力
            if extension_counts:
                self.log_message("拡張子別発見ファイル数:")
                total_raw = 0
                for ext, count in extension_counts.items():
                    self.log_message(f"  {ext}: {count}個")
                    total_raw += count
                
                self.log_message(f"生ファイル数合計: {total_raw}個")
                self.log_message(f"重複排除後: {len(image_files)}個")
                
                if total_raw != len(image_files):
                    duplicates = total_raw - len(image_files)
                    self.log_message(f"重複ファイル除去: {duplicates}個")
            
            if not image_files:
                self.log_message("処理対象の画像ファイルが見つかりません")
                self.log_message(f"検索パス確認: {Path(input_folder).exists()}")
                
                # フォルダ内容の詳細確認
                try:
                    folder_contents = list(Path(input_folder).iterdir())
                    self.log_message(f"フォルダ内総アイテム数: {len(folder_contents)}")
                    self.log_message(f"最初の10個: {[item.name for item in folder_contents[:10]]}")
                    
                    # サブフォルダ確認
                    subfolders = [item for item in folder_contents if item.is_dir()]
                    if subfolders:
                        self.log_message(f"サブフォルダ数: {len(subfolders)}")
                        self.log_message(f"サブフォルダ: {[sf.name for sf in subfolders[:5]]}")
                        
                except Exception as e:
                    self.log_message(f"フォルダ内容確認エラー: {e}")
                
                messagebox.showinfo("情報", "処理対象の画像ファイルが見つかりません")
                return
            
            # 統計設定
            self.stats['total_files'] = len(image_files)
            self.log_message(f"✅ 最終対象ファイル数: {self.stats['total_files']}")
            self.log_message(f"最初の5ファイル: {[f.name for f in image_files[:5]]}")
            
            # ファイルサイズ統計（デバッグ用）
            try:
                total_size = sum(f.stat().st_size for f in image_files if f.exists())
                avg_size = total_size / len(image_files) if image_files else 0
                self.log_message(f"総ファイルサイズ: {total_size / (1024*1024):.1f}MB")
                self.log_message(f"平均ファイルサイズ: {avg_size / 1024:.1f}KB")
            except Exception as e:
                self.log_message(f"ファイルサイズ統計エラー: {e}")
            
            # 各画像を分類・移動
            for i, image_file in enumerate(image_files):
                try:
                    # プログレス更新
                    progress = (i / len(image_files)) * 100
                    self.progress_var.set(progress)
                    self.status_var.set(f"処理中... ({i+1}/{len(image_files)}) {image_file.name}")
                    
                    # 分類実行
                    self.log_message(f"--- ファイル処理開始: {image_file.name} ---")
                    predicted_class, confidence = self.classify_image(image_file)
                    
                    if predicted_class is None:
                        self.log_message(f"🔴 分類失敗: {image_file.name} - predicted_classがNone")
                        self.stats['error_files'] += 1
                        continue
                    
                    # 🎯 **信頼度チェック強化** 🎯
                    current_threshold = self.confidence_threshold
                    self.log_message(f"🔍 信頼度チェック: {confidence:.4f} vs 闾値{current_threshold:.2f}")
                    
                    if confidence < current_threshold:
                        self.log_message(f"⚠️  **低信頼度スキップ**: {image_file.name}")
                        self.log_message(f"   予測: {predicted_class} (信頼度: {confidence:.4f})")
                        self.log_message(f"   闾値: {current_threshold:.2f} → スキップ処理実行")
                        
                        # スキップ処理
                        skip_success = self._handle_low_confidence_file(
                            image_file, predicted_class, confidence, output_folder
                        )
                        
                        if skip_success:
                            self.stats['low_confidence_skipped'] += 1
                            self.stats['skipped_files'].append({
                                'filename': image_file.name,
                                'predicted_class': predicted_class,
                                'confidence': confidence,
                                'threshold': current_threshold
                            })
                        else:
                            self.stats['error_files'] += 1
                        
                        continue  # 次のファイルへ
                    
                    # 新しいファイル名生成
                    new_filename = self.generate_new_filename(image_file, predicted_class)
                    
                    # 移動先パス
                    dest_folder = Path(output_folder) / predicted_class
                    dest_path = dest_folder / new_filename
                    
                    # 重複ファイル名対応
                    counter = 1
                    original_dest_path = dest_path
                    while dest_path.exists():
                        stem = original_dest_path.stem
                        suffix = original_dest_path.suffix
                        dest_path = original_dest_path.parent / f"{stem}_{counter}{suffix}"
                        counter += 1
                    
                    # ファイル移動またはコピー
                    if self.copy_files_var.get():
                        shutil.copy2(image_file, dest_path)
                        action = "コピー"
                    else:
                        shutil.move(str(image_file), dest_path)
                        action = "移動"
                    
                    self.log_message(f"{action}: {image_file.name} → {predicted_class}/{dest_path.name} (信頼度: {confidence:.3f})")
                    
                    # 統計更新
                    self.stats['classified'][predicted_class] += 1
                    self.stats['moved_files'] += 1
                    if new_filename != image_file.name:
                        self.stats['renamed_files'] += 1
                    
                except Exception as e:
                    self.log_message(f"ファイル処理エラー: {image_file.name} - {str(e)}")
                    self.stats['error_files'] += 1
                
                self.stats['processed_files'] += 1
                
                # UI更新（10枚ごと）
                if i % 10 == 0:
                    self.update_stats_display()
            
            # 処理完了
            self.progress_var.set(100)
            self.status_var.set("処理完了")
            self.update_stats_display()
            
            # 結果サマリー（詳細統計）
            self.log_message("\n=== 処理結果サマリー ===")
            self.log_message(f"📊 ファイル数統計:")
            self.log_message(f"  - 検出総ファイル数: {self.stats['total_files']}")
            self.log_message(f"  - 実際処理数: {self.stats['processed_files']}")
            self.log_message(f"  - 成功移動数: {self.stats['moved_files']}")
            self.log_message(f"  - 低信頼度スキップ数: {self.stats['low_confidence_skipped']}")
            self.log_message(f"  - リネーム数: {self.stats['renamed_files']}")
            self.log_message(f"  - エラー数: {self.stats['error_files']}")
            
            # 処理率計算
            if self.stats['total_files'] > 0:
                process_rate = (self.stats['processed_files'] / self.stats['total_files']) * 100
                success_rate = (self.stats['moved_files'] / self.stats['processed_files']) * 100 if self.stats['processed_files'] > 0 else 0
                skip_rate = (self.stats['low_confidence_skipped'] / self.stats['processed_files']) * 100 if self.stats['processed_files'] > 0 else 0
                
                self.log_message(f"📈 処理率: {process_rate:.1f}% ({self.stats['processed_files']}/{self.stats['total_files']})")
                self.log_message(f"📈 成功率: {success_rate:.1f}% ({self.stats['moved_files']}/{self.stats['processed_files']})")
                self.log_message(f"📈 スキップ率: {skip_rate:.1f}% ({self.stats['low_confidence_skipped']}/{self.stats['processed_files']})")
            
            # 信頼度統計
            if self.stats['skipped_files']:
                self.log_message(f"\n🎯 信頼度統計 (閾値: {self.confidence_threshold:.2f}):")
                confidences = [item['confidence'] for item in self.stats['skipped_files']]
                self.log_message(f"  - スキップファイル最高信頼度: {max(confidences):.4f}")
                self.log_message(f"  - スキップファイル最低信頼度: {min(confidences):.4f}")
                self.log_message(f"  - スキップファイル平均信頼度: {sum(confidences)/len(confidences):.4f}")
                
                # スキップファイル例
                self.log_message(f"  - スキップファイル例:")
                for i, skip_info in enumerate(self.stats['skipped_files'][:3]):
                    self.log_message(f"    {i+1}. {skip_info['filename']} ({skip_info['predicted_class']}: {skip_info['confidence']:.4f})")
            
            # クラス別統計
            self.log_message(f"🎯 クラス別分類結果:")
            classified_total = sum(self.stats['classified'].values())
            for class_name, count in self.stats['classified'].items():
                percentage = (count / max(1, classified_total)) * 100
                pile_code = self.pile_codes.get(class_name, '?')
                self.log_message(f"  - {class_name}({pile_code}): {count}枚 ({percentage:.1f}%)")
            
            # 整合性チェック
            expected_total = self.stats['moved_files'] + self.stats['error_files'] + self.stats['low_confidence_skipped']
            if self.stats['processed_files'] != expected_total:
                self.log_message(f"⚠️  統計整合性警告: 処理数{self.stats['processed_files']} ≠ 移動+エラー+スキップ{expected_total}")
            
            if classified_total != self.stats['moved_files']:
                self.log_message(f"⚠️  分類統計警告: 分類総数{classified_total} ≠ 移動数{self.stats['moved_files']}")
            
            messagebox.showinfo("完了", 
                              f"分類・整理が完了しました。\n\n"
                              f"📁 検出ファイル数: {self.stats['total_files']}枚\n"
                              f"⚙️ 実際処理数: {self.stats['processed_files']}枚\n"
                              f"✅ 成功移動数: {self.stats['moved_files']}枚\n"
                              f"⚠️ 低信頼度スキップ: {self.stats['low_confidence_skipped']}枚\n"
                              f"❌ エラー数: {self.stats['error_files']}枚")
            
        except Exception as e:
            self.log_message(f"処理エラー: {str(e)}")
            self.status_var.set("処理エラー")
            messagebox.showerror("エラー", f"処理中にエラーが発生しました:\n{str(e)}")
        
        finally:
            self.execute_button.config(state="normal")

def main():
    """メイン関数"""
    root = tk.Tk()
    app = CoreClassifierOrganizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()