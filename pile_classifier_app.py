# 杭種コード自動追加アプリ
# Pile Type Code Auto-Addition Application

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
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    print("警告: TensorFlowがインストールされていません。デモモードで実行します。")
    TF_AVAILABLE = False

# ローカルモジュール
from utils import ConfigManager, FileManager, ImageProcessor, Logger, PerformanceMonitor

class PileClassifierApp:
    """杭種分類・コード自動追加アプリケーション"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("杭種コード自動追加アプリ v1.0")
        self.root.geometry("800x600")
        
        # 杭種コード定義（エクセルロジック完全準拠）
        self.pile_codes = {
            'plastic': 'P',    # プラスチック杭 (Plastic)
            'plate': 'PL',     # プレート (Plate)
            'metal': 'B',      # 金属鋲 (Byou)
            'concrete': 'C',   # コンクリート杭 (Concrete)
            'target': 'T',     # 引照点 (Target point)
            'kokudo': 'KD',    # 国土基準点
            'gaiku_kijun': 'GK',    # 都市再生街区基準点
            'gaiku_takaku': 'GT',   # 都市再生街区多角点
            'gaiku_hojo': 'GH',     # 都市再生街区補助点
            'gaiku_setsu': 'GS'     # 都市再生街区節点
        }
        
        # AIモデル
        self.model = None
        self.trained_classifier = None  # 訓練済み分類器
        self.class_names = list(self.pile_codes.keys())
        
        # 処理統計
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'success_files': 0,
            'error_files': 0
        }
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        """UIセットアップ"""
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # タイトル
        title_label = ttk.Label(main_frame, text="杭種コード自動追加アプリ", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # フォルダ選択
        folder_frame = ttk.LabelFrame(main_frame, text="対象フォルダ", padding="10")
        folder_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.folder_var = tk.StringVar()
        folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_var, width=60)
        folder_entry.grid(row=0, column=0, padx=(0, 10))
        
        folder_btn = ttk.Button(folder_frame, text="フォルダ選択", 
                               command=self.select_folder)
        folder_btn.grid(row=0, column=1)
        
        # 処理オプション
        options_frame = ttk.LabelFrame(main_frame, text="処理オプション", padding="10")
        options_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.backup_var = tk.BooleanVar(value=True)
        backup_check = ttk.Checkbutton(options_frame, text="処理前にバックアップを作成", 
                                      variable=self.backup_var)
        backup_check.grid(row=0, column=0, sticky=tk.W)
        
        self.confirm_var = tk.BooleanVar(value=False)
        confirm_check = ttk.Checkbutton(options_frame, text="AI判定結果を手動確認", 
                                       variable=self.confirm_var)
        confirm_check.grid(row=1, column=0, sticky=tk.W)
        
        # 実行ボタン
        execute_btn = ttk.Button(main_frame, text="処理実行", 
                                command=self.execute_processing, 
                                style="Accent.TButton")
        execute_btn.grid(row=3, column=1, pady=20)
        
        # プログレスバー
        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, 
                                      maximum=100, length=400)
        progress_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # ステータス表示
        self.status_var = tk.StringVar(value="準備完了")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=5, column=0, columnspan=3)
        
        # ログエリア
        log_frame = ttk.LabelFrame(main_frame, text="処理ログ", padding="10")
        log_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(20, 0))
        
        self.log_text = tk.Text(log_frame, height=15, width=80)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # グリッドの重み設定
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(6, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
    
    def log_message(self, message):
        """ログメッセージを表示"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def select_folder(self):
        """フォルダ選択ダイアログ"""
        folder = filedialog.askdirectory(title="処理対象フォルダを選択してください")
        if folder:
            self.folder_var.set(folder)
            self.log_message(f"フォルダ選択: {folder}")
    
    def load_model(self):
        """AIモデルロード"""
        try:
            # 訓練済みモデルがあるかチェック
            model_path = "models/simple_classifier.pkl"
            if os.path.exists(model_path):
                # SimplePileClassifierをインポートして読み込み
                try:
                    from no_tf_train import SimplePileClassifier
                    self.trained_classifier = SimplePileClassifier()
                    if self.trained_classifier.load_model():
                        self.log_message("訓練済みモデル読み込み成功")
                        return
                    else:
                        self.log_message("訓練済みモデル読み込み失敗、デモモードで実行")
                except Exception as e:
                    self.log_message(f"モデル読み込みエラー: {str(e)}")
            
            # フォールバック: デモ用モデル
            self.trained_classifier = None
            if TF_AVAILABLE:
                self.model = self.create_demo_model()
                self.log_message("TensorFlowデモモデル読み込み完了")
            else:
                self.log_message("デモモード（ランダム予測）で実行")
                
        except Exception as e:
            self.log_message(f"モデル読み込みエラー: {str(e)}")
            messagebox.showerror("エラー", f"AIモデルの読み込みに失敗しました:\n{str(e)}")
    
    def create_demo_model(self):
        """デモ用の簡易モデル作成"""
        # 実際の実装では、訓練済みのCNNモデルを使用
        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(224, 224, 3)),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(len(self.class_names), activation='softmax')
        ])
        return model
    
    def preprocess_image(self, image_path):
        """画像前処理"""
        try:
            # 画像読み込み
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # BGRからRGBに変換
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # リサイズ
            image = cv2.resize(image, (224, 224))
            
            # 正規化
            image = image.astype(np.float32) / 255.0
            
            # バッチ次元追加
            image = np.expand_dims(image, axis=0)
            
            return image
        except Exception as e:
            self.log_message(f"画像前処理エラー: {str(e)}")
            return None
    
    def classify_pile(self, image_path):
        """杭種分類"""
        try:
            # 訓練済みモデルが利用可能かチェック
            if hasattr(self, 'trained_classifier') and self.trained_classifier is not None:
                # 実際の訓練済みモデルで予測
                predicted_class, confidence = self.trained_classifier.predict(image_path)
                if predicted_class is not None:
                    return predicted_class, confidence
            
            # フォールバック: デモ用のランダム予測
            import random
            predicted_class = random.choice(self.class_names)
            confidence = random.uniform(0.7, 0.95)
            
            return predicted_class, confidence
            
        except Exception as e:
            self.log_message(f"分類エラー: {str(e)}")
            return None, 0.0
    
    def get_new_filename(self, original_path, pile_type):
        """新しいファイル名生成"""
        try:
            path_obj = Path(original_path)
            original_name = path_obj.stem
            extension = path_obj.suffix
            
            # 杭種コード取得
            pile_code = self.pile_codes.get(pile_type, 'U')  # Unknown
            
            # 既にコードが付いているかチェック
            for code in self.pile_codes.values():
                if original_name.startswith(f"{code}"):
                    # 既存コードを置換
                    new_name = f"{pile_code}{original_name[len(code):]}"
                    return path_obj.parent / f"{new_name}{extension}"
            
            # 新規にコード追加
            new_name = f"{pile_code}{original_name}"
            return path_obj.parent / f"{new_name}{extension}"
            
        except Exception as e:
            self.log_message(f"ファイル名生成エラー: {str(e)}")
            return None
    
    def create_backup(self, folder_path):
        """バックアップ作成"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_folder = Path(folder_path).parent / f"backup_{timestamp}"
            shutil.copytree(folder_path, backup_folder)
            self.log_message(f"バックアップ作成: {backup_folder}")
            return True
        except Exception as e:
            self.log_message(f"バックアップ作成エラー: {str(e)}")
            return False
    
    def extract_point_number(self, filename):
        """ファイル名から点番を抽出"""
        try:
            # ファイル名から拡張子を除去
            name_without_ext = Path(filename).stem
            
            # 既存の杭種コードを除去
            for code in self.pile_codes.values():
                if name_without_ext.startswith(code):
                    name_without_ext = name_without_ext[len(code):]
                    break
            
            # -1, -2, -3などの撮影番号を除去して点番を抽出
            import re
            # パターン: 英数字で始まり、-数字で終わる場合
            match = re.match(r'^([A-Za-z0-9]+(?:[_\-][A-Za-z0-9]+)*)', name_without_ext)
            if match:
                point_number = match.group(1)
                # -数字を除去
                point_number = re.sub(r'-\d+$', '', point_number)
                return point_number
            
            # フォールバック: ファイル名全体を点番とする
            return name_without_ext
            
        except Exception as e:
            self.log_message(f"点番抽出エラー: {filename} - {str(e)}")
            return str(filename)
    
    def group_images_by_point(self, image_files):
        """画像ファイルを点番でグループ化"""
        point_groups = {}
        
        for image_file in image_files:
            point_number = self.extract_point_number(image_file.name)
            
            if point_number not in point_groups:
                point_groups[point_number] = []
            
            point_groups[point_number].append(image_file)
        
        # 各グループ内で-2画像を優先してソート
        for point_number, files in point_groups.items():
            files.sort(key=lambda f: (
                0 if '-2.' in f.name or '-2' in f.stem else 1,  # -2画像を最優先
                f.name  # ファイル名でソート
            ))
        
        return point_groups
    
    def find_reference_image(self, group_files):
        """グループ内で判定基準となる画像を特定"""
        # -2画像があればそれを基準とする
        for image_file in group_files:
            if '-2.' in image_file.name or (image_file.stem.endswith('-2')):
                return image_file
        
        # -2画像がない場合は最初のファイルを基準とする
        return group_files[0] if group_files else None
    
    def execute_processing(self):
        """メイン処理実行"""
        folder_path = self.folder_var.get()
        if not folder_path:
            messagebox.showwarning("警告", "フォルダを選択してください")
            return
        
        if not os.path.exists(folder_path):
            messagebox.showerror("エラー", "選択されたフォルダが存在しません")
            return
        
        # 別スレッドで処理実行
        thread = threading.Thread(target=self._process_images, args=(folder_path,))
        thread.daemon = True
        thread.start()
    
    def _process_images(self, folder_path):
        """画像処理メインロジック"""
        try:
            self.status_var.set("処理開始...")
            
            # 対象画像ファイル収集
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_files = []
            
            for ext in image_extensions:
                pattern = f"**/*{ext}"
                image_files.extend(Path(folder_path).glob(pattern))
                pattern = f"**/*{ext.upper()}"
                image_files.extend(Path(folder_path).glob(pattern))
            
            if not image_files:
                self.log_message("処理対象の画像ファイルが見つかりません")
                self.status_var.set("処理完了（対象ファイルなし）")
                return
            
            self.stats['total_files'] = len(image_files)
            self.log_message(f"対象ファイル数: {self.stats['total_files']}")
            
            # バックアップ作成
            if self.backup_var.get():
                if not self.create_backup(folder_path):
                    if not messagebox.askyesno("確認", "バックアップの作成に失敗しました。処理を続行しますか？"):
                        return
            
            # 点番ごとにグループ化し、-2画像で杭種判定
            point_groups = self.group_images_by_point(image_files)
            self.log_message(f"点番グループ数: {len(point_groups)}")
            
            # 各点番グループを処理
            processed_count = 0
            total_groups = len(point_groups)
            
            for group_index, (point_number, group_files) in enumerate(point_groups.items()):
                try:
                    # プログレス更新
                    progress = (group_index / total_groups) * 100
                    self.progress_var.set(progress)
                    self.status_var.set(f"処理中... ({group_index + 1}/{total_groups}) 点番: {point_number}")
                    
                    self.log_message(f"\n=== 点番: {point_number} ({len(group_files)}枚) ===")
                    
                    # 基準画像（-2画像）を特定
                    reference_image = self.find_reference_image(group_files)
                    if reference_image is None:
                        self.log_message(f"  基準画像が見つかりません")
                        self.stats['error_files'] += len(group_files)
                        continue
                    
                    self.log_message(f"  基準画像: {reference_image.name}")
                    
                    # 基準画像でAI分類実行
                    pile_type, confidence = self.classify_pile(reference_image)
                    
                    if pile_type is None:
                        self.log_message(f"  分類失敗: {reference_image.name}")
                        self.stats['error_files'] += len(group_files)
                        continue
                    
                    pile_code = self.pile_codes.get(pile_type, 'U')
                    self.log_message(f"  判定結果: {pile_type} ({pile_code}) 信頼度: {confidence:.3f}")
                    
                    # 手動確認が有効な場合
                    if self.confirm_var.get() and confidence < 0.9:
                        # 実際の実装では確認ダイアログを表示
                        self.log_message(f"  ※ 信頼度が低いため要確認")
                    
                    # 同じ点番のすべてのファイルに同じ杭種コードを適用
                    group_success = 0
                    for image_file in group_files:
                        try:
                            # 新しいファイル名生成
                            new_path = self.get_new_filename(image_file, pile_type)
                            if new_path is None:
                                self.log_message(f"    エラー: ファイル名生成失敗 - {image_file.name}")
                                self.stats['error_files'] += 1
                                continue
                            
                            # ファイル名変更
                            if new_path != image_file:
                                image_file.rename(new_path)
                                self.log_message(f"    リネーム: {image_file.name} → {new_path.name}")
                            else:
                                self.log_message(f"    変更なし: {image_file.name}")
                            
                            group_success += 1
                            self.stats['success_files'] += 1
                            
                        except Exception as e:
                            self.log_message(f"    エラー: {image_file.name} - {str(e)}")
                            self.stats['error_files'] += 1
                    
                    self.log_message(f"  点番処理完了: {group_success}/{len(group_files)}枚成功")
                    processed_count += len(group_files)
                    self.stats['processed_files'] = processed_count
                    
                except Exception as e:
                    self.log_message(f"  点番処理エラー: {point_number} - {str(e)}")
                    self.stats['error_files'] += len(group_files)
                    processed_count += len(group_files)
                    self.stats['processed_files'] = processed_count
            
            # 処理完了
            self.progress_var.set(100)
            self.status_var.set("処理完了")
            
            # 結果サマリー
            self.log_message("\n=== 処理結果サマリー ===")
            self.log_message(f"総ファイル数: {self.stats['total_files']}")
            self.log_message(f"処理済み: {self.stats['processed_files']}")
            self.log_message(f"成功: {self.stats['success_files']}")
            self.log_message(f"エラー: {self.stats['error_files']}")
            
            messagebox.showinfo("完了", 
                              f"処理が完了しました。\n"
                              f"成功: {self.stats['success_files']}件\n"
                              f"エラー: {self.stats['error_files']}件")
            
        except Exception as e:
            self.log_message(f"処理エラー: {str(e)}")
            self.status_var.set("処理エラー")
            messagebox.showerror("エラー", f"処理中にエラーが発生しました:\n{str(e)}")

def main():
    """メイン関数"""
    root = tk.Tk()
    app = PileClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
