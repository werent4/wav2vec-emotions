from transformers import AutoTokenizer, AutoConfig
from gliclass import GLiClassModelConfig, GLiClassModel
from gliclass.pipeline import BaseZeroShotClassificationPipeline, AudioEncoderZeroShotClassificationPipeline, ZeroShotClassificationPipeline
import torch, torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from gliclass.data_processing import GLiClassDataset
from dataclasses import dataclass
import json
import os
import random
import numpy as np
from tqdm import tqdm
import warnings
from pprint import pprint
from dataset import EMOTION2ID_GLICLASS
random.seed(42)


@dataclass
class EvaluationObject:
    name: str
    model: Wav2Vec2ForSequenceClassification
    audio_feature_extractor: Wav2Vec2FeatureExtractor
    
    def get_name(self):
        return self.name
    
    def get_model(self):
        return self.model
    
    def get_features_extractor(self):
        return self.audio_feature_extractor
    

class Evaluaor:
    def __init__(
        self,
        dataset_name: str,
        loader_fn: callable,
        models_names: list,
        eval_subset_size: int = 5000,
        max_length_seconds: int = 5
    ) -> None:
        self.dataset = self.load_dataset(dataset_name, loader_fn, eval_subset_size)
        self.evaluation_objects = self.load_models(models_names)
        self.max_length_seconds = max_length_seconds

        self.all_labels = list(EMOTION2ID_GLICLASS.keys())
        self.label2id = EMOTION2ID_GLICLASS
        self.id2label = {v: k for k, v in EMOTION2ID_GLICLASS.items()}

    def load_dataset(self, dataset_name: str, loader_fn, eval_subset_size):
        dataset = loader_fn(dataset_name)
        return dataset[:eval_subset_size]

    # def collect_all_labels(self):
    #     all_labels = set()
    #     for row in self.dataset:
    #         if 'all_labels' in row:
    #             all_labels.update(row['all_labels'])
    #         else:
    #             warnings.warn(f"missing collumn all_labels for {row['id']} row", UserWarning)
    #     print("Tottal unique labels: ", len(all_labels))
    #     return list(all_labels)

    def load_models(self, models_names):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        evaluation_objects = []
        for model_name in models_names:
            model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name).to(self.device)
            audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                model_name,
            )

            evaluation_objects.append(
                EvaluationObject(
                    name= model_name,
                    model= model,
                    audio_feature_extractor=audio_feature_extractor
                )
            )

        return evaluation_objects

    def calculate_results(self, logits, true_labels):
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        
        tp = {label: 0 for label in self.all_labels}
        fp = {label: 0 for label in self.all_labels}
        tn = {label: 0 for label in self.all_labels}
        fn = {label: 0 for label in self.all_labels}

        for pred_id, true_id in zip(predictions, true_labels):
            pred_label = self.id2label[pred_id]
            true_label = self.id2label[true_id]
            
            for label in self.all_labels:
                if label == pred_label and label == true_label:
                    tp[label] += 1
                elif label == pred_label and label != true_label:
                    fp[label] += 1
                elif label != pred_label and label == true_label:
                    fn[label] += 1
                else:
                    tn[label] += 1
        
        return tp, fp, tn, fn
    
    def calculate_metrics(self, tp, fp, tn, fn):
        precision = {}
        recall = {}
        f1 = {}
        
        for label in self.all_labels:
            # Precision: TP / (TP + FP)
            precision[label] = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0
            
            # Recall: TP / (TP + FN)
            recall[label] = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0
            
            # F1-score: 2 * (precision * recall) / (precision + recall)
            f1[label] = 2 * (precision[label] * recall[label]) / (precision[label] + recall[label]) if (precision[label] + recall[label]) > 0 else 0

        macro_precision = sum(precision.values()) / len(precision)
        macro_recall = sum(recall.values()) / len(recall)
        macro_f1 = sum(f1.values()) / len(f1)

        total_tp = sum(tp.values())
        total_fp = sum(fp.values())
        total_tn = sum(tn.values())
        total_fn = sum(fn.values())

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        accuracy = total_tp / (total_tp + total_fp + total_tn + total_fn) if (total_tp + total_fp + total_tn + total_fn) > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "accuracy": accuracy
        }

    def get_clean_results(self, full_results):
        clean_results = {}
        for model_name, results in full_results.items():
            clean_results[model_name] = {
                "macro_precision": results["macro_precision"],
                "macro_recall": results["macro_recall"],
                "macro_f1": results["macro_f1"],
                "micro_precision": results["micro_precision"],
                "micro_recall": results["micro_recall"],
                "micro_f1": results["micro_f1"],
                "accuracy": results["accuracy"],
            }
        return clean_results

    def prepare_audio_batch(self, audio_paths, sample_rates, feature_extractor):
        audio_arrays = []
        
        for audio_path, sample_rate in zip(audio_paths, sample_rates):
            audio_array = torch.load(audio_path).float()
            
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, 
                    new_freq=16000
                )
                audio_array = resampler(audio_array)
            
            audio_arrays.append(audio_array.numpy())
        
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=16000 * self.max_length_seconds
        )
        
        return inputs

    def evaluate(self, batch_size: int = 8, return_clean = True):
        models_results = {}
        for object_ in self.evaluation_objects:
            name = object_.get_name()
            model = object_.get_model()
            audio_feature_extractor = object_.get_features_extractor()

            print("Evaluating model: ", name)
            model.eval()
            models_results[name]= {
                "tp" : {label: 0 for label in self.all_labels},
                "fp" : {label: 0 for label in self.all_labels},
                "tn" : {label: 0 for label in self.all_labels},
                "fn" : {label: 0 for label in self.all_labels}
            }

            for i in tqdm(range(0, len(self.dataset), batch_size), desc= "Evaluating"):
                batch_rows = self.dataset[i:i+batch_size]

                audio_paths = [item['audio_path'] for item in batch_rows]
                sample_rates = [item['sample_rate'] for item in batch_rows]
                true_labels = [item['true_label'] for item in batch_rows]

                inputs = self.prepare_audio_batch(audio_paths, sample_rates, audio_feature_extractor)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}


                with torch.no_grad():
                    logits = model(**inputs).logits
                batch_tp, batch_fp, batch_tn, batch_fn = self.calculate_results(logits, true_labels)
                for label in self.all_labels:
                    models_results[name]["tp"][label] += batch_tp[label]
                    models_results[name]["fp"][label] += batch_fp[label]
                    models_results[name]["tn"][label] += batch_tn[label]
                    models_results[name]["fn"][label] += batch_fn[label]

            metrics = self.calculate_metrics(
                models_results[name]["tp"],
                models_results[name]["fp"],
                models_results[name]["tn"],
                models_results[name]["fn"]
            )
            models_results[name].update(metrics)
            if return_clean:
                pprint(self.get_clean_results(models_results))

            if return_clean:
                models_results = self.get_clean_results(models_results)
        return models_results

def load_emotions_dataset(dataset_name, dataset_path: str = "/home/werent4/wav2vec-emotions/datasets/eval_emotions.json"):
    with open(dataset_path, "r", encoding='utf-8') as f:
        dataset = json.load(f)
    random.shuffle(dataset)
    new_rows = []
    for row in dataset:
        all_labels = [EMOTION2ID_GLICLASS[label.lower()] for label in row["all_labels"]] 
        label = [EMOTION2ID_GLICLASS[label.lower()] for label in row["true_labels"]] 
        if len(label) > 1:
            raise ValueError("ONLY ONE LABEL")
        new_row = {
            "id": row["id"], 
            "audio_path": row["audio_path"], 
            "sample_rate": row["sample_rate"], 
            "all_labels": all_labels, 
            "true_label": label[0]
        }
        new_rows.append(new_row)
    return new_rows

######### RUNNERS FUNC ################

def eval_emotions(eval_size):
    DATASET_NAME = "Hemg/Emotion-audio-Dataset"
    evaluator = Evaluaor(
        dataset_name= DATASET_NAME,
        loader_fn = load_emotions_dataset,
        models_names= [
            "/home/werent4/wav2vec-emotions/python/output_model/best_f1"
        ],
        eval_subset_size= eval_size,

    )
    results = evaluator.evaluate(batch_size= 4)
    print("Results for: ", DATASET_NAME)
    pprint(results)


def main():
    EVAL_SIZE = 5000
    eval_emotions(EVAL_SIZE)    


if __name__ == "__main__":
    main()