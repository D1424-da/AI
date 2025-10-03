# メモリ最適化版杭種分類訓練スクリプト
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import gc
import psutil
from datetime import datetime
from config_loader import ConfigLoader
import tkinter as tk
from tkinter import filedialog, messagebox

class MemoryOptimizedPileClassifierTrainer:
    """メモリ最適化版杭種分類モデル訓練クラス"""
    
    def __init__(self, data_dir=None, model_save_path="models/all_pile_classifier.h5", config_path="config.json"):
        # データディレクトリが指定されていない場合は選択ダイアログを表示
        if data_dir is None:
            data_dir = self.select_training_data_directory()
            if not data_dir:
                raise ValueError("訓練データディレクトリが選択されませんでした")
        
        self.data_dir = Path(data_dir)
        self.model_save_path = model_save_path
        
        # 設定ファイル読み込み
        self.config = ConfigLoader(config_path)
        
        # 設定から値を取得（フォールバック付き）
        self.image_size = self.config.image_size
        self.batch_size = self.config.batch_size  
        self.epochs = self.config.epochs
        
        # クラス定義（設定ファイルから）
        self.class_names = self.config.class_order
        if not self.class_names:
            # フォールバック: 12クラス定義
            self.class_names = [
                'plastic', 'plate', 'byou', 'concrete',
                'traverse', 'kokudo', 'gaiku_sankaku', 'gaiku_setsu',
                'gaiku_takaku', 'gaiku_hojo', 'traverse_in', 'kagoshima_in'
            ]
        
        self.label_encoder = LabelEncoder()
        self.model = None
        
        # 設定情報表示
        print(f"🔧 設定情報:")
        print(f"   クラス数: {len(self.class_names)}")
        print(f"   クラス順序: {self.class_names}")
        print(f"   画像サイズ: {self.image_size}")
        print(f"   バッチサイズ: {self.batch_size}")
        print(f"   エポック数: {self.epochs}")
        print(f"   学習率: {self.config.learning_rate}")
        print(f"   メモリ最適化: {self.config.memory_optimization}")
        
        # メモリ最適化設定
        self.enable_memory_optimization()
    
    def select_training_data_directory(self):
        """訓練データディレクトリ選択ダイアログ"""
        root = tk.Tk()
        root.withdraw()  # メインウィンドウを非表示
        
        messagebox.showinfo(
            "訓練データディレクトリ選択",
            "訓練データが格納されているディレクトリを選択してください。\n\n"
            "ディレクトリ構造:\n"
            "選択ディレクトリ/\n"
            "  ├── plastic/\n"
            "  ├── plate/\n"
            "  ├── byou/\n"
            "  ├── concrete/\n"
            "  └── その他のクラスフォルダ..."
        )
        
        data_dir = filedialog.askdirectory(
            title="訓練データディレクトリを選択",
            initialdir=os.getcwd()
        )
        
        root.destroy()
        
        if data_dir:
            print(f"✅ 選択された訓練データディレクトリ: {data_dir}")
            # ディレクトリ構造を確認
            self.verify_directory_structure(data_dir)
            return data_dir
        else:
            print("❌ ディレクトリが選択されませんでした")
            return None
    
    def verify_directory_structure(self, data_dir):
        """ディレクトリ構造の確認"""
        data_path = Path(data_dir)
        subdirs = [d for d in data_path.iterdir() if d.is_dir()]
        
        print(f"\n📁 発見されたサブディレクトリ:")
        for subdir in subdirs:
            image_count = len(list(subdir.glob('*.jpg')) + list(subdir.glob('*.jpeg')) + 
                            list(subdir.glob('*.png')) + list(subdir.glob('*.bmp')))
            print(f"   {subdir.name}: {image_count}枚")
        
        if not subdirs:
            messagebox.showwarning(
                "警告", 
                f"選択されたディレクトリ内にサブディレクトリが見つかりません:\n{data_dir}\n\n"
                "各クラス名のサブディレクトリが必要です。"
            )
    
    def enable_memory_optimization(self):
        """TensorFlowメモリ最適化設定"""
        # メモリ増加を段階的に設定
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        # CPU使用時の最適化
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    
    def check_memory_usage(self):
        """メモリ使用率チェック"""
        memory = psutil.virtual_memory()
        usage_percent = memory.percent
        available_gb = memory.available / (1024**3)
        
        print(f"メモリ使用率: {usage_percent:.1f}% (利用可能: {available_gb:.1f}GB)")
        
        if usage_percent > 90:
            print("警告: メモリ使用率が危険レベルです")
            return False
        return True
    
    def memory_efficient_image_generator(self, image_files, labels, batch_size):
        """メモリ効率的な画像ジェネレーター"""
        while True:
            # バッチごとにシャッフル
            indices = np.random.permutation(len(image_files))
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_images = []
                batch_labels = []
                
                for idx in batch_indices:
                    # 画像を即座に処理・正規化
                    image = self.load_and_preprocess_image_efficient(image_files[idx])
                    if image is not None:
                        batch_images.append(image)
                        batch_labels.append(labels[idx])
                
                if batch_images:
                    # NumPy配列に変換（メモリ効率重視）
                    X_batch = np.array(batch_images, dtype=np.float32)
                    y_batch = np.array(batch_labels)
                    
                    # 即座にメモリ開放
                    del batch_images
                    gc.collect()
                    
                    yield X_batch, y_batch
    
    def load_and_preprocess_image_efficient(self, image_path):
        """メモリ効率重視の画像読み込み"""
        try:
            # PILを使用してメモリ効率改善
            from PIL import Image
            with Image.open(image_path) as img:
                # RGB変換
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # リサイズ（高品質）
                img = img.resize(self.image_size, Image.LANCZOS)
                
                # NumPy配列に変換・正規化
                image_array = np.array(img, dtype=np.float32) / 255.0
                
                return image_array
        
        except Exception as e:
            print(f"画像読み込みエラー: {image_path} - {str(e)}")
            return None
    
    def prepare_file_lists_only(self):
        """ファイルパスのみを準備（画像は読み込まない）"""
        print("ファイルリスト準備中...")
        
        image_files = []
        labels = []
        class_counts = {}
        
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"警告: {class_dir} が見つかりません")
                continue
            
            # 画像ファイルパスのみ収集
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            class_files = []
            
            for ext in extensions:
                class_files.extend(list(class_dir.glob(ext)))
            
            class_counts[class_name] = len(class_files)
            print(f"{class_name}: {len(class_files)}枚")
            
            # ファイルパスとラベルを追加
            image_files.extend(class_files)
            labels.extend([class_name] * len(class_files))
        
        if not image_files:
            raise ValueError("画像ファイルが見つかりません")
        
        # ラベルエンコーディング（順序を保持）
        # 🚨 CRITICAL: LabelEncoderは自動ソートするため、手動マッピングで順序保持
        class_to_index = {class_name: i for i, class_name in enumerate(self.class_names)}
        encoded_labels = [class_to_index[label] for label in labels]
        
        # LabelEncoderを正しい順序で初期化（推論時の互換性のため）
        self.label_encoder.classes_ = np.array(self.class_names)
        
        categorical_labels = keras.utils.to_categorical(encoded_labels, len(self.class_names))
        
        print(f"総ファイル数: {len(image_files)}")
        print("クラス分布:", class_counts)
        print(f"🔍 定義されたクラス順序: {self.class_names}")
        print(f"🔍 LabelEncoder順序: {self.label_encoder.classes_.tolist()}")
        print(f"🔍 クラス→インデックスマッピング: {class_to_index}")
        
        return image_files, categorical_labels
    
    def create_memory_efficient_model(self):
        """メモリ効率重視のモデル作成"""
        print("メモリ効率モデル作成中...")
        
        # 軽量なベースモデル使用
        base_model = keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.image_size, 3),
            alpha=0.75  # モデルサイズ削減
        )
        
        # 特徴抽出のみ（メモリ節約）
        base_model.trainable = False
        
        # シンプルな分類ヘッド
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.25),
            layers.Dense(128, activation='relu'),  # 層を小さく
            layers.Dropout(0.25),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        # 軽量オプティマイザ（設定ファイルから学習率取得）
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"モデルパラメータ数: {model.count_params():,}")
        return model
    
    def train_with_generator(self, train_files, train_labels, val_files, val_labels):
        """ジェネレーターを使用した訓練"""
        print("ジェネレーター訓練開始...")
        
        # ステップ数計算
        train_steps = len(train_files) // self.batch_size
        val_steps = len(val_files) // self.batch_size
        
        print(f"訓練ステップ/エポック: {train_steps}")
        print(f"検証ステップ/エポック: {val_steps}")
        
        # ジェネレーター作成
        train_gen = self.memory_efficient_image_generator(
            train_files, train_labels, self.batch_size
        )
        val_gen = self.memory_efficient_image_generator(
            val_files, val_labels, self.batch_size
        )
        
        # コールバック（メモリ監視付き）
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                self.model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            # カスタムメモリ監視コールバック
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: self.memory_callback(epoch, logs)
            )
        ]
        
        # 訓練実行
        history = self.model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            epochs=self.epochs,
            validation_data=val_gen,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def memory_callback(self, epoch, logs):
        """メモリ使用量監視コールバック"""
        if epoch % 5 == 0:  # 5エポックごと
            memory = psutil.virtual_memory()
            print(f"エポック {epoch+1} - メモリ使用率: {memory.percent:.1f}%")
            
            if memory.percent > 95:
                print("警告: メモリ使用率が危険レベル。ガベージコレクション実行")
                gc.collect()
    
    def train_optimized(self):
        """最適化された訓練実行"""
        try:
            print("=== メモリ最適化訓練開始 ===")
            
            # 初期メモリチェック
            if not self.check_memory_usage():
                print("警告: 既にメモリ使用率が高すぎます")
            
            # ファイルリスト準備（画像は読み込まない）
            image_files, labels = self.prepare_file_lists_only()
            
            # 訓練・検証分割
            train_files, val_files, train_labels, val_labels = train_test_split(
                image_files, labels, test_size=0.2, random_state=42,
                stratify=np.argmax(labels, axis=1)
            )
            
            print(f"訓練ファイル: {len(train_files)}")
            print(f"検証ファイル: {len(val_files)}")
            
            # メモリ効率モデル作成
            self.model = self.create_memory_efficient_model()
            
            # ジェネレーター訓練
            history = self.train_with_generator(
                train_files, train_labels, val_files, val_labels
            )
            
            # 結果表示
            self.plot_results(history)
            
            # モデル情報保存
            self.save_model_info()
            
            print("訓練完了!")
            return True
            
        except Exception as e:
            print(f"訓練エラー: {str(e)}")
            return False
        finally:
            # 強制ガベージコレクション
            gc.collect()
    
    def plot_results(self, history):
        """結果可視化"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='訓練精度')
        plt.plot(history.history['val_accuracy'], label='検証精度')
        plt.title('精度推移')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='訓練損失')
        plt.plot(history.history['val_loss'], label='検証損失')
        plt.title('損失推移')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_results.png')
        plt.show()
    
    def save_model_info(self):
        """モデル情報保存（詳細版）"""
        model_info = {
            'model_type': 'all_pile_classifier',
            'class_names': self.class_names,  # 定義順序
            'image_size': list(self.image_size),
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.config.learning_rate,
            'target_accuracy': 0.90,
            'model_save_path': self.model_save_path,
            'memory_optimized': True,
            'total_classes': len(self.class_names),
            'label_encoder_classes': self.label_encoder.classes_.tolist(),  # 実際の順序
            'training_timestamp': datetime.now().isoformat()
        }
        
        # 🚨 順序整合性の検証
        print(f"🔍 クラス順序検証:")
        print(f"   定義順序: {self.class_names}")
        print(f"   Encoder順序: {self.label_encoder.classes_.tolist()}")
        
        if self.class_names != self.label_encoder.classes_.tolist():
            print("⚠️  警告: クラス順序が不一致！")
            print("   推論時に誤分類が発生する可能性があります")
        else:
            print("✅ クラス順序一致確認")
        
        # メインの model_info.json を保存
        info_path = Path(self.model_save_path).parent / "model_info.json"
        os.makedirs(Path(self.model_save_path).parent, exist_ok=True)
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        # 12クラス専用の情報ファイルも保存
        all_pile_info_path = Path(self.model_save_path).parent / "all_pile_model_info.json"
        with open(all_pile_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        print(f"✅ モデル情報保存完了:")
        print(f"   {info_path}")
        print(f"   {all_pile_info_path}")

# 使用例
if __name__ == "__main__":
    try:
        # 訓練データディレクトリを対話的に選択
        print("=== AI杭種分類モデル訓練 ===")
        print("訓練データディレクトリを選択してください...")
        
        trainer = MemoryOptimizedPileClassifierTrainer()  # data_dirを省略すると選択ダイアログが表示される
        
        # 訓練実行
        success = trainer.train_optimized()
        
        if success:
            print("\n🎉 訓練が正常に完了しました！")
            print(f"モデル保存先: {trainer.model_save_path}")
        else:
            print("\n❌ 訓練中にエラーが発生しました")
            
    except Exception as e:
        print(f"\nエラー: {str(e)}")
        input("Enterキーを押して終了...")
    
    input("\nEnterキーを押して終了...")
    