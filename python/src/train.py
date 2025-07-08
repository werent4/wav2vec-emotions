import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datasets import load_dataset, Dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, TrainingArguments
import json
import os
import pprint

from dataset import (
    IEMOCAPDataset,
    GLiClassEmotionAudioDataset,
    DataCollatorForWav2Vec2Classification,
    EMOTION2ID, SIMPLIFIED_EMOTION2ID, EMOTION2ID_GLICLASS,
    print_emotion_distribution
)
from loss_function import FocalLoss
from _trainer import LossTrainer

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    print("Loaded configuration:")
    pprint.pprint(config)
    return config

def load_preprocessor_and_model(model_name: str, device: str = "cuda", label2id: dict = EMOTION2ID_GLICLASS):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label={i: label for label, i in label2id.items()},
        label2id=label2id
    ).to(device)
    return feature_extractor, model

def computeFalse_loss(model, inputs, loss_fn):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    return loss_fn(outputs.logits, labels)

def compute_metrics(pred):

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        preds, 
        average='weighted',  
        zero_division=0  
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def save_config(config, eval_results):
    save_path = os.path.join(config["training_args"]["output_dir"], "training_config.json")
    full_config = {
        "model_name": config["model_name"],
        "emotion_mapping": EMOTION2ID,
        "max_length_seconds": config.get("max_length_seconds", 5),
        "mapping_type": config.get("mapping_type", False),
        "training_args": config["training_args"],
        "eval_results": eval_results
    }

    with open(save_path, "w") as f:
        json.dump(full_config, f, indent=4)

    print(f"Configuration saved to {save_path}")

def main(args):
    config = load_config(args.config)

    mapping_type = config.get("mapping_type", "gliclass")
    if mapping_type == "simple":
        emotion2id = SIMPLIFIED_EMOTION2ID
    elif mapping_type == "gliclass":
        emotion2id = EMOTION2ID_GLICLASS
    else:
        emotion2id = EMOTION2ID

    if args.model_name:
        config["model_name"] = args.model_name
    if args.output_dir:
        config["training_args"]["output_dir"] = args.output_dir


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor, model = load_preprocessor_and_model(
        config["model_name"], 
        device, 
        label2id=emotion2id
    )

    max_length_s= config.get("max_lenght_seconds", 5)

    emo_train = GLiClassEmotionAudioDataset(
        dataset_path=config["dataset_path"],
        feature_extractor=feature_extractor,
        emotion_mapping=emotion2id,
        sampling_rate= config['model_sampling_rate'],
        max_length_s= max_length_s
    )

    emo_val = GLiClassEmotionAudioDataset(
        dataset_path= "/home/werent4/wav2vec-emotions/datasets/eval_emotions.json",
        feature_extractor=feature_extractor,
        emotion_mapping=emotion2id,
        sampling_rate= config['model_sampling_rate'],
        max_length_s= max_length_s
    )

    data_collator = DataCollatorForWav2Vec2Classification(
        feature_extractor=feature_extractor,
        padding=True,
        max_length=feature_extractor.sampling_rate * max_length_s,
        return_tensors="pt"
    )

    training_args = TrainingArguments(
        **config["training_args"]
    )

    focal_loss = FocalLoss(gamma=config.get("gamma", 2.0), alpha=config.get("alpha", 0.5)).to(device)
    trainer = LossTrainer(
        model=model,
        args=training_args,
        train_dataset=emo_train,
        eval_dataset=emo_val, 
        compute_metrics=compute_metrics,  
        data_collator = data_collator,
        loss_fn=focal_loss
    )
    trainer.train()

    best_f1_path = os.path.join(training_args.output_dir, 'best_f1')
    trainer.save_model(output_dir=best_f1_path)
    print(f"Best model saved to {best_f1_path}")

    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    save_config(config, eval_results)

    feature_extractor.save_pretrained(best_f1_path)
    print(f"Feature extractor saved to {best_f1_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a Wav2Vec2 model for emotion recognition.")
    parser.add_argument("--config", type=str, default="config.json", 
                       help="Path to JSON configuration file")
    parser.add_argument("--model_name", type=str, 
                       help="Model name from Hugging Face Hub (overrides config)")
    parser.add_argument("--output_dir", type=str, 
                       help="Output directory for model (overrides config)")
    args = parser.parse_args()

    main(args)