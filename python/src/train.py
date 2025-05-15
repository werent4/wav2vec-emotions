import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datasets import load_dataset, Dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, TrainingArguments
import json
import os
import pprint

from dataset import IEMOCAPDataset, EMOTION2ID
from loss_function import FocalLoss
from _trainer import LossTrainer

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    print("Loaded configuration:")
    pprint.pprint(config)
    return config

def load_preprocessor_and_model(model_name: str, device: str = "cuda", label2id: dict = EMOTION2ID):
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
        "is_simplified": config.get("is_simplified", False),
        "training_args": config["training_args"],
        "eval_results": eval_results
    }

    with open(save_path, "w") as f:
        json.dump(full_config, f, indent=4)

    print(f"Configuration saved to {save_path}")

def main(args):
    config = load_config(args.config)

    if args.model_name:
        config["model_name"] = args.model_name
    if args.output_dir:
        config["training_args"]["output_dir"] = args.output_dir

    dataset = load_dataset("AbstractTTS/IEMOCAP", trust_remote_code= True).shuffle(seed=42)
    train_indices, val_indices = train_test_split(
        range(len(dataset["train"])), 
        test_size=config.get("val_size", 0.15), 
        random_state=42, 
        stratify=[item["major_emotion"] for item in dataset["train"]]
    )

    train_dataset_split = dataset["train"].select(train_indices)
    val_dataset_split = dataset["train"].select(val_indices)
    print(f"Train indices: {train_dataset_split}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor, model = load_preprocessor_and_model(
        config["model_name"], 
        device, 
        label2id=EMOTION2ID
    )

    max_length= feature_extractor.sampling_rate * config.get("max_lenght_seconds", 5)

    iemocap_train = IEMOCAPDataset(
        dataset=train_dataset_split,
        emotion_mapping=EMOTION2ID,
        feature_extractor=feature_extractor,
        is_simplified=config.get("is_simplified", False),
        max_length=max_length,
        use_augmentations=False
    )

    iemocap_val = IEMOCAPDataset(
        dataset=val_dataset_split,
        emotion_mapping=EMOTION2ID,
        feature_extractor=feature_extractor,
        is_simplified=config.get("is_simplified", False),
        max_length=max_length,
        use_augmentations=False
    )

    training_args = TrainingArguments(
        **config["training_args"]
    )

    focal_loss = FocalLoss(gamma=config.get("gamma", 2.0), alpha=config.get("alpha", 0.1))
    trainer = LossTrainer(
        model=model,
        args=training_args,
        train_dataset=iemocap_train,
        eval_dataset=iemocap_val, 
        compute_metrics=compute_metrics,  
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