import argparse
import os
import torch
import json
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

SUPPORTED_EXPORT_TYPES = ["onnx"]

def load_preprocessor_and_model(model_path: str):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    return feature_extractor, model

def export_to_onnx(model, feature_extractor, output_path, sample_rate=16000, duration=5):
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        raise ImportError("The ONNX package is required for ONNX export. Install it with: pip install onnx onnxruntime")
    
    os.makedirs(output_path, exist_ok=True)
    model.eval()
    
    max_length = sample_rate * duration
    dummy_input = torch.randn(1, max_length)
    
    input_names = ["input"]
    output_names = ["logits"]
    
    dynamic_axes = {
        "input": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"}
    }
    
    onnx_path = os.path.join(output_path, "model.onnx")
    
    try:
        torch.onnx.export(
            model,                      
            dummy_input,                
            onnx_path,                  
            input_names=input_names,    
            output_names=output_names,  
            dynamic_axes=dynamic_axes,  
            opset_version=14,           
            do_constant_folding=True,   
            export_params=True,         
            verbose=True                
        )
        
        print(f"ONNX model saved to: {onnx_path}")
        
        print("Checking ONNX model validity...")
        imported_model = onnx.load(onnx_path)
        onnx.checker.check_model(imported_model)
        print("ONNX model is valid!")
        
    except Exception as e:
        print(f"ONNX export failed with error: {str(e)}")
        print("\nTry running with a different opset version, e.g., --opset_version 13")
        return None, None
    
    print(f"ONNX model saved to: {onnx_path}")
    
    metadata = {
        "id2label": model.config.id2label,
        "sample_rate": feature_extractor.sampling_rate,
        "max_length": max_length,
        "framework": "onnx"
    }
    
    metadata_path = os.path.join(output_path, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Metadata saved to: {metadata_path}")
    
    # test the ONNX model
    try:
        session = ort.InferenceSession(onnx_path)
        
        test_input = np.random.randn(1, max_length).astype(np.float32)
        
        ort_inputs = {session.get_inputs()[0].name: test_input}
        ort_outputs = session.run(None, ort_inputs)
        
        print(f"ONNX model successfully verified! Output size: {ort_outputs[0].shape}")
        print(f"Number of emotion classes: {ort_outputs[0].shape[1]}")
        print("Test prediction probabilities:", ort_outputs[0][0][:5], "...")

    except Exception as e:
        print(f"ONNX runtime test failed with error: {str(e)}")
    
    return onnx_path, metadata_path

def main(args):   
    feature_extractor, model = load_preprocessor_and_model(args.model_path)
    if args.export_type == "onnx":
        print("Exporting to ONNX format...")
        export_to_onnx(
            model=model,
            feature_extractor=feature_extractor,
            output_path=args.output_path,
            sample_rate=args.sample_rate,
            duration=args.duration
        )
    else:
        raise NotImplementedError(f"Export type {args.export_type} is not implemented yet. Supported types: {SUPPORTED_EXPORT_TYPES}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Wav2Vec2 emotion model to ONNX")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained PyTorch model")
    parser.add_argument("--output_path", type=str, default="onnx_model", help="Output directory for ONNX model")
    parser.add_argument("--export_type", type=str, choices=SUPPORTED_EXPORT_TYPES, default="onnx", help="Export type")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--duration", type=int, default=5, help="Maximum audio duration in seconds")
    
    args = parser.parse_args()
    main(args)