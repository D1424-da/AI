"""
段階的学習ユーティリティ - 13万枚対応
Progressive Training Utilities for Ultra-Large Datasets
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import gc
import psutil
from pathlib import Path


class ProgressiveStageGenerator(keras.utils.Sequence):
    """段階的学習用ジェネレータ"""
    
    def __init__(self, base_generator, subset_size, batch_size, stage_name="stage"):
        self.base_generator = base_generator
        self.subset_size = min(subset_size, len(base_generator.filepaths))
        self.batch_size = batch_size
        self.stage_name = stage_name
        
        # サブセットインデックス選択（バランス考慮）
        self.indices = self._select_balanced_subset()
        
        print(f"   📊 {stage_name}: {self.subset_size:,}枚選択 (batch_size={batch_size})")
        
    def _select_balanced_subset(self):
        """クラスバランスを考慮したサブセット選択"""
        try:
            # 各クラスから均等に選択
            all_labels = np.argmax(self.base_generator.labels, axis=1)
            unique_classes = np.unique(all_labels)
            samples_per_class = self.subset_size // len(unique_classes)
            
            selected_indices = []
            
            for class_id in unique_classes:
                class_indices = np.where(all_labels == class_id)[0]
                
                if len(class_indices) > samples_per_class:
                    # ランダムサンプリング
                    selected = np.random.choice(
                        class_indices, 
                        samples_per_class, 
                        replace=False
                    )
                else:
                    # 全て選択
                    selected = class_indices
                    
                selected_indices.extend(selected)
            
            # 不足分を追加
            remaining = self.subset_size - len(selected_indices)
            if remaining > 0:
                all_indices = set(range(len(self.base_generator.filepaths)))
                unused_indices = list(all_indices - set(selected_indices))
                
                if unused_indices:
                    additional = np.random.choice(
                        unused_indices,
                        min(remaining, len(unused_indices)),
                        replace=False
                    )
                    selected_indices.extend(additional)
            
            return np.array(selected_indices[:self.subset_size])
            
        except Exception as e:
            print(f"   ⚠️  バランス選択失敗、ランダム選択: {e}")
            return np.random.choice(
                len(self.base_generator.filepaths),
                self.subset_size,
                replace=False
            )
    
    def __len__(self):
        return int(np.ceil(self.subset_size / self.batch_size))
    
    def __getitem__(self, idx):
        """バッチデータ取得"""
        batch_start = idx * self.batch_size
        batch_end = min(batch_start + self.batch_size, self.subset_size)
        batch_indices = self.indices[batch_start:batch_end]
        
        # ベースジェネレータからデータ取得
        images = []
        labels = []
        
        for idx in batch_indices:
            try:
                # ベースジェネレータの画像・ラベル取得
                img_path = self.base_generator.filepaths[idx]
                label = self.base_generator.labels[idx]
                
                # 画像読み込み（PIL使用）
                img = self.base_generator.load_and_preprocess_image(img_path)
                
                if img is not None:
                    images.append(img)
                    labels.append(label)
                    
            except Exception as e:
                print(f"   ⚠️  画像読み込みエラー: {e}")
                continue
        
        if not images:
            # エラー時はダミーデータ返却
            return (
                np.zeros((1, *self.base_generator.image_size, 3)),
                np.zeros((1, len(self.base_generator.class_names)))
            )
        
        return np.array(images), np.array(labels)
    
    def on_epoch_end(self):
        """エポック終了時の処理"""
        # メモリクリーンアップ
        if hasattr(self, '_cleanup_counter'):
            self._cleanup_counter += 1
        else:
            self._cleanup_counter = 1
            
        if self._cleanup_counter % 3 == 0:  # 3エポックごと
            gc.collect()
            
            memory = psutil.virtual_memory()
            if memory.percent > 85:
                print(f"   🧹 メモリクリーンアップ: {memory.percent:.1f}% → ", end="")
                gc.collect()
                tf.keras.backend.clear_session()
                memory_after = psutil.virtual_memory()
                print(f"{memory_after.percent:.1f}%")


class UltraLargeDatasetTrainer:
    """13万枚対応の段階的学習トレーナー"""
    
    def __init__(self, base_trainer):
        self.base_trainer = base_trainer
        self.progressive_stages = None
        
    def setup_progressive_stages(self, total_images):
        """段階的学習ステージ設定"""
        if total_images > 100000:
            # 13万枚対応の3段階学習
            self.progressive_stages = {
                'foundation': {
                    'subset': 25000,
                    'epochs': 15,
                    'lr': 0.001,
                    'batch': 64,
                    'description': '基礎学習（2.5万枚）'
                },
                'expansion': {
                    'subset': 70000,
                    'epochs': 12,
                    'lr': 0.0003,
                    'batch': 96,
                    'description': '拡張学習（7万枚）'
                },
                'refinement': {
                    'subset': total_images,
                    'epochs': 10,
                    'lr': 0.00005,
                    'batch': 128,
                    'description': f'高精度学習（{total_images:,}枚）'
                }
            }
        elif total_images > 50000:
            # 5-10万枚用の2段階学習
            self.progressive_stages = {
                'foundation': {
                    'subset': total_images // 2,
                    'epochs': 20,
                    'lr': 0.001,
                    'batch': 64,
                    'description': f'基礎学習（{total_images//2:,}枚）'
                },
                'refinement': {
                    'subset': total_images,
                    'epochs': 15,
                    'lr': 0.0001,
                    'batch': 96,
                    'description': f'高精度学習（{total_images:,}枚）'
                }
            }
        else:
            # 通常学習
            return False
            
        print(f"\n🎯 段階的学習ステージ設定完了:")
        for stage_name, params in self.progressive_stages.items():
            print(f"   {stage_name}: {params['description']}")
            
        return True
    
    def execute_progressive_training(self, train_generator, val_generator):
        """段階的学習実行"""
        if not self.progressive_stages:
            return None
            
        print("\n🚀 段階的学習開始")
        all_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        for stage_name, params in self.progressive_stages.items():
            print(f"\n📈 {stage_name.upper()}: {params['description']}")
            
            # 段階別ジェネレータ作成
            stage_generator = ProgressiveStageGenerator(
                train_generator,
                params['subset'],
                params['batch'],
                stage_name
            )
            
            # 学習率調整
            if hasattr(self.base_trainer.model.optimizer, 'learning_rate'):
                self.base_trainer.model.optimizer.learning_rate.assign(params['lr'])
                print(f"   📊 学習率設定: {params['lr']}")
            
            # 段階学習実行
            stage_history = self.base_trainer.model.fit(
                stage_generator,
                validation_data=val_generator,
                epochs=params['epochs'],
                callbacks=self.base_trainer.get_callbacks(),
                verbose=1
            )
            
            # 履歴統合
            for metric in all_history.keys():
                if metric in stage_history.history:
                    all_history[metric].extend(stage_history.history[metric])
            
            # ステージ結果表示
            final_acc = stage_history.history.get('val_accuracy', [0])[-1]
            print(f"   ✅ {stage_name}完了: val_accuracy={final_acc:.4f}")
            
            # メモリ最適化
            gc.collect()
            tf.keras.backend.clear_session()
        
        # 疑似historyオブジェクト作成
        return type('History', (), {'history': all_history})()
    
    def validate_ultra_dataset(self, total_images):
        """超大規模データセット検証"""
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        print(f"\n🔍 超大規模データセット検証:")
        print(f"   データセットサイズ: {total_images:,}枚")
        print(f"   利用可能メモリ: {available_gb:.1f}GB")
        
        # メモリ要件推定
        estimated_memory_gb = (total_images * 640 * 640 * 3 * 4) / (1024**3)  # float32
        
        if estimated_memory_gb > available_gb * 0.8:
            print(f"   ⚠️  推定メモリ要件: {estimated_memory_gb:.1f}GB")
            print(f"   🎯 段階的学習が必要です")
            return True
        else:
            print(f"   ✅ 通常学習で処理可能")
            return False