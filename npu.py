import onnxruntime as ort

# DirectML provider確認
providers = ort.get_available_providers()
print(f"利用可能プロバイダー: {providers}")

if 'DmlExecutionProvider' in providers:
    print("AMD NPU/GPU利用可能（DirectML経由）")
else:
    print("DirectML未対応")