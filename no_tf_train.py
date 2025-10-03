# TensorFlow無しでも動作する杭種分類器
# Pile Classifier without TensorFlow dependency

import os
import numpy as np
from pathlib import Path
import json
from PIL import Image
import pickle
from datetime import datetime

# scikit-learnを使用した機械学習
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, accuracy_score
    SKLEARN_AVAILABLE = True
    print("scikit-learn が利用可能です")
except ImportError:
    print("警告: scikit-learn がインストールされていません")
    SKLEARN_AVAILABLE = False

class SimplePileClassifier:
    """TensorFlow無しの簡易杭種分類器"""
    
    def __init__(self, data_dir="training_data", model_save_path="models/simple_classifier.pkl"):
        self.data_dir = Path(data_dir)
        self.model_save_path = model_save_path
        self.image_size = (64, 64)  # 特徴抽出用の小さなサイズ
        
        # 杭種クラス定義
        self.class_names = [
            'plastic',      # プラスチック杭
            'plate',        # プレート
            'metal',        # 金属杭
            'concrete',     # コンクリート杭
            'target',       # 引照点
            'kokudo',       # 国土基準点
            'gaiku_kijun',  # 都市再生街区基準点
            'gaiku_takaku', # 都市再生街区多角点
            'gaiku_hojo',   # 都市再生街区補助点
            'gaiku_setsu'   # 都市再生街区節点
        ]
        
        self.model = None
        self.label_encoder = LabelEncoder() if SKLEARN_AVAILABLE else None
        
    def extract_color_features(self, image_array):
        """色特徴を抽出"""
        # RGB平均値
        rgb_mean = np.mean(image_array, axis=(0, 1))
        
        # RGB標準偏差
        rgb_std = np.std(image_array, axis=(0, 1))
        
        # HSV変換
        try:
            import cv2
            hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
            hsv_mean = np.mean(hsv, axis=(0, 1))
        except:
            hsv_mean = np.array([0, 0, 0])
        
        # 特徴ベクトルを結合
        features = np.concatenate([rgb_mean, rgb_std, hsv_mean])
        return features
    
    def extract_texture_features(self, image_array):
        """テクスチャ特徴を抽出"""
        # グレースケール変換
        gray = np.mean(image_array, axis=2)
        
        # 簡単な統計的特徴
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        min_val = np.min(gray)
        max_val = np.max(gray)
        
        # エッジ特徴（簡易版）
        try:
            import cv2
            edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
            edge_density = np.sum(edges > 0) / edges.size
        except:
            edge_density = 0
        
        return np.array([mean_val, std_val, min_val, max_val, edge_density])
    
    def extract_features(self, image_array):
        """総合特徴抽出"""
        color_features = self.extract_color_features(image_array)
        texture_features = self.extract_texture_features(image_array)
        
        # 特徴を結合
        features = np.concatenate([color_features, texture_features])
        return features
    
    def load_and_preprocess_image(self, image_path):
        """画像読み込み・前処理"""
        try:
            # Pillowで画像読み込み
            image = Image.open(image_path)
            
            # RGBモードに変換
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # リサイズ
            image = image.resize(self.image_size)
            
            # NumPy配列に変換
            image_array = np.array(image)
            
            return image_array
            
        except Exception as e:
            print(f"画像読み込みエラー: {image_path} - {str(e)}")
            return None
    
    def prepare_dataset(self):
        """データセット準備"""
        print("データセット準備中...")
        
        features_list = []
        labels = []
        
        # 各クラスフォルダから画像を読み込み
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"警告: クラスフォルダが見つかりません: {class_dir}")
                continue
            
            image_files = list(class_dir.glob("*.jpg")) + \
                         list(class_dir.glob("*.jpeg")) + \
                         list(class_dir.glob("*.png"))
            
            print(f"{class_name}: {len(image_files)}枚")
            
            for image_file in image_files:
                image_array = self.load_and_preprocess_image(image_file)
                if image_array is not None:
                    features = self.extract_features(image_array)
                    features_list.append(features)
                    labels.append(class_name)
        
        if not features_list:
            raise ValueError("画像データが見つかりません。データフォルダ構造を確認してください。")
        
        # NumPy配列に変換
        X = np.array(features_list)
        y = np.array(labels)
        
        print(f"総画像数: {len(X)}")
        print(f"特徴次元: {X.shape[1]}")
        print(f"クラス数: {len(self.class_names)}")
        
        return X, y
    
    def create_model(self):
        """機械学習モデル作成"""
        if not SKLEARN_AVAILABLE:
            print("scikit-learnが利用できないため、モデル作成をスキップします")
            return None
            
        print("ランダムフォレスト分類器を作成中...")
        
        # ランダムフォレスト分類器
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        return model
    
    def train_model(self, X_train, y_train):
        """モデル訓練"""
        if not SKLEARN_AVAILABLE or self.model is None:
            print("モデルが利用できないため、訓練をスキップします")
            return
            
        print("モデル訓練開始...")
        
        # ラベルエンコーディング
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # 訓練実行
        self.model.fit(X_train, y_train_encoded)
        
        print("訓練完了")
    
    def evaluate_model(self, X_test, y_test):
        """モデル評価"""
        if not SKLEARN_AVAILABLE or self.model is None:
            print("モデルが利用できないため、評価をスキップします")
            return None
            
        print("\nモデル評価中...")
        
        # ラベルエンコーディング
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # 予測
        y_pred = self.model.predict(X_test)
        
        # 精度計算
        accuracy = accuracy_score(y_test_encoded, y_pred)
        print(f"テスト精度: {accuracy:.4f}")
        
        # クラス別精度
        print("\nクラス別レポート:")
        print(classification_report(y_test_encoded, y_pred, 
                                  target_names=self.class_names))
        
        return accuracy
    
    def save_model(self):
        """モデル保存"""
        if self.model is None:
            print("保存するモデルがありません")
            return
            
        # モデル保存ディレクトリ作成
        model_dir = Path(self.model_save_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # モデルとラベルエンコーダーを保存
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'class_names': self.class_names,
            'image_size': self.image_size,
            'feature_dim': None  # 後で設定
        }
        
        with open(self.model_save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"モデル保存: {self.model_save_path}")
    
    def load_model(self):
        """モデル読み込み"""
        if not os.path.exists(self.model_save_path):
            print(f"モデルファイルが見つかりません: {self.model_save_path}")
            return False
            
        try:
            with open(self.model_save_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.class_names = model_data['class_names']
            self.image_size = model_data['image_size']
            
            print(f"モデル読み込み完了: {self.model_save_path}")
            return True
            
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            return False
    
    def predict(self, image_path):
        """単一画像の予測"""
        if self.model is None:
            return None, 0.0
            
        # 画像前処理
        image_array = self.load_and_preprocess_image(image_path)
        if image_array is None:
            return None, 0.0
        
        # 特徴抽出
        features = self.extract_features(image_array)
        features = features.reshape(1, -1)
        
        # 予測
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        # クラス名と信頼度
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        confidence = np.max(probabilities)
        
        return predicted_class, confidence
    
    def train(self):
        """訓練実行メイン"""
        try:
            print("=== 簡易杭種分類器訓練開始 ===")
            print(f"使用手法: Random Forest (scikit-learn)")
            
            # データセット準備
            X, y = self.prepare_dataset()
            
            # データ分割
            if SKLEARN_AVAILABLE:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                # 簡易分割
                split = int(len(X) * 0.8)
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]
            
            print(f"訓練データ: {X_train.shape[0]}")
            print(f"テストデータ: {X_test.shape[0]}")
            print(f"特徴次元: {X_train.shape[1]}")
            
            # モデル作成
            self.model = self.create_model()
            
            if self.model is not None:
                # 訓練実行
                self.train_model(X_train, y_train)
                
                # 評価
                accuracy = self.evaluate_model(X_test, y_test)
                
                # モデル保存
                self.save_model()
                
                print(f"\n訓練完了！精度: {accuracy:.4f}")
                print(f"モデル保存場所: {self.model_save_path}")
            else:
                print("モデル作成ができませんでした")
            
            print("=== 訓練処理完了 ===")
            
        except Exception as e:
            print(f"訓練エラー: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """メイン関数"""
    print("簡易杭種分類器訓練スクリプト v1.0")
    print("=" * 50)
    
    # 依存関係チェック
    if not SKLEARN_AVAILABLE:
        print("scikit-learnがインストールされていません。")
        print("以下のコマンドでインストールしてください:")
        print("pip install scikit-learn")
        return
    
    # 訓練実行
    classifier = SimplePileClassifier()
    classifier.train()
    
    # 簡単なテスト
    print("\n" + "=" * 50)
    print("モデルテスト")
    
    # テスト用画像でのテスト
    test_dir = Path("training_data/plastic")
    if test_dir.exists():
        test_images = list(test_dir.glob("*.jpg"))
        if test_images:
            test_image = test_images[0]
            predicted_class, confidence = classifier.predict(test_image)
            print(f"テスト画像: {test_image.name}")
            print(f"予測結果: {predicted_class} (信頼度: {confidence:.3f})")

if __name__ == "__main__":
    main()
