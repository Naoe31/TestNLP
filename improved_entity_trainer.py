"""
Enhanced NER trainer with improved entity-specific handling and balanced data generation.
Focuses on improving performance for challenging entities like MATERIAL and LUMBAR_SUPPORT.
"""

import spacy
import json
import os
import random
import time
from typing import List, Tuple, Dict, Set
from collections import defaultdict
from datetime import datetime
import numpy as np
from spacy.training import Example

# Fixed random seed for reproducibility
RANDOM_SEED = 42

class EntityBalancer:
    """Handles entity-specific data balancing and augmentation strategies"""
    
    def __init__(self, base_synonyms: Dict[str, List[str]]):
        self.base_synonyms = base_synonyms
        self.entity_stats = {}
        self.weak_entities = set()
        
    def analyze_entity_distribution(self, training_data: List[Tuple]):
        """Analyze entity distribution and identify weak entities"""
        entity_counts = defaultdict(int)
        entity_contexts = defaultdict(list)
        
        for text, annotations in training_data:
            for start, end, label in annotations.get("entities", []):
                entity_counts[label] += 1
                # Store surrounding context
                context_start = max(0, start - 50)
                context_end = min(len(text), end + 50)
                entity_contexts[label].append({
                    'text': text[start:end],
                    'context': text[context_start:context_end]
                })
        
        # Calculate statistics
        total_entities = sum(entity_counts.values())
        for entity, count in entity_counts.items():
            self.entity_stats[entity] = {
                'count': count,
                'percentage': count / total_entities * 100,
                'contexts': entity_contexts[entity]
            }
            
            # Identify weak entities (less than 5% of total or less than 10 examples)
            if count < 10 or (count / total_entities) < 0.05:
                self.weak_entities.add(entity)
        
        return self.entity_stats
    
    def generate_balanced_examples(self, entity: str, count: int) -> List[Tuple]:
        """Generate balanced examples for a specific entity"""
        examples = []
        synonyms = self.base_synonyms.get(entity, [])
        
        if entity in self.weak_entities:
            # Generate more diverse examples for weak entities
            count = int(count * 1.5)  # 50% more examples for weak entities
        
        templates = self._get_entity_templates(entity)
        for _ in range(count):
            example_data = self._create_example(entity, templates, synonyms)
            if example_data:
                text, spans = example_data
                examples.append((text, {"entities": [(start, end, entity) for start, end in spans]}))
        
        return examples
    
    def _get_entity_templates(self, entity: str) -> List[str]:
        """Get entity-specific templates"""
        base_templates = {
            "MATERIAL": [
                "The seat is made of {value} material",
                "The {value} upholstery feels comfortable",
                "High-quality {value} used in construction",
                "Premium {value} covers the entire seat",
                "Soft and durable {value} material",
                "The seat features {value} covering"
            ],
            "LUMBAR_SUPPORT": [
                "The {value} provides excellent back support",
                "Adjustable {value} for comfort",
                "Built-in {value} helps with posture",
                "Ergonomic design with {value}",
                "The seat includes a {value} feature",
                "Comfortable {value} for long sitting"
            ]
        }
        
        return base_templates.get(entity, [
            "The {value} is well designed",
            "Comfortable {value} feature",
            "The {value} works perfectly",
            "High-quality {value}",
            "Advanced {value} system"
        ])
    
    def _create_example(self, entity: str, templates: List[str], synonyms: List[str]) -> Tuple[str, List[Tuple]]:
        """Create a single synthetic example"""
        template = random.choice(templates)
        synonym = random.choice(synonyms)
        text = template.format(value=synonym)
        
        # Find the position of the entity in the text
        start = text.find(synonym)
        end = start + len(synonym)
        
        return text, [(start, end)]

class ImprovedNERTrainer:
    """Enhanced NER trainer with better handling of challenging entities"""
    
    def __init__(self, train_data: List[Tuple], balancer: EntityBalancer):
        self.train_data = train_data
        self.balancer = balancer
        self.nlp = spacy.blank("en")
        self.metrics_history = []
        self.best_model_state = None
        self.best_f1 = 0.0
        
        # Entity-specific hyperparameters
        self.entity_hyperparams = {
            "MATERIAL": {"dropout": 0.3, "batch_size": 16},
            "LUMBAR_SUPPORT": {"dropout": 0.3, "batch_size": 16},
            "default": {"dropout": 0.2, "batch_size": 32}
        }
    
    def create_stratified_split(self, min_examples_per_entity: int = 10) -> Tuple[List[Example], List[Example]]:
        """Create a stratified split ensuring minimum examples per entity"""
        entity_examples = defaultdict(list)
        all_examples = []
        
        # Group examples by entity
        for text, annotations in self.train_data:
            doc = self.nlp.make_doc(text)
            entities = annotations.get("entities", [])
            example = Example.from_dict(doc, {"entities": entities})
            
            entities_in_example = {ent[2] for ent in entities}
            for entity in entities_in_example:
                entity_examples[entity].append(example)
            all_examples.append(example)
        
        train_examples = []
        val_examples = []
        
        # Stratified split for each entity
        for entity, examples in entity_examples.items():
            random.shuffle(examples)
            val_size = max(min_examples_per_entity, int(len(examples) * 0.2))
            val_examples.extend(examples[:val_size])
            train_examples.extend(examples[val_size:])
        
        # Remove duplicates while preserving order
        train_examples = list(dict.fromkeys(train_examples))
        val_examples = list(dict.fromkeys(val_examples))
        
        return train_examples, val_examples
    
    def train(self, n_iter: int = 100) -> spacy.language.Language:
        """Train the model with enhanced entity handling"""
        spacy.util.fix_random_seed(RANDOM_SEED)
        
        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe("ner")
        
        # Add labels
        for _, annotations in self.train_data:
            for _, _, label in annotations.get("entities", []):
                if label not in ner.labels:
                    ner.add_label(label)
        
        train_examples, val_examples = self.create_stratified_split()
        
        # Initialize the model
        self.nlp.initialize(lambda: train_examples)
        
        # Training loop with entity-specific handling
        patience = 0
        max_patience = 15
        best_f1 = 0.0
        
        for epoch in range(n_iter):
            # Shuffle with fixed seed for reproducibility
            random.shuffle(train_examples)
            
            losses = {}
            batches = self._create_smart_batches(train_examples, epoch)
            
            # Training step
            for batch in batches:
                try:
                    losses.update(self.nlp.update(
                        batch,
                        drop=self._get_dropout_rate(batch, epoch),
                        losses=losses
                    ))
                except Exception as e:
                    print(f"Error during training: {e}")
                    continue
            
            # Evaluation
            if val_examples:
                metrics = self._evaluate(val_examples)
                current_f1 = metrics['entity_level']['f1']
                
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    self.best_model_state = self.nlp.to_bytes()
                    patience = 0
                else:
                    patience += 1
                
                # Early stopping
                if patience >= max_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
                
                # Log progress
                if (epoch + 1) % 10 == 0:
                    self._log_progress(epoch + 1, losses, metrics)
        
        # Restore best model
        if self.best_model_state:
            self.nlp.from_bytes(self.best_model_state)
        
        return self.nlp
    
    def _create_smart_batches(self, examples: List[Example], epoch: int) -> List[List[Example]]:
        """Create smart batches with entity-specific handling"""
        if epoch < 20:
            # Focus on challenging entities in early epochs
            batches = []
            challenging_examples = [ex for ex in examples if self._has_challenging_entity(ex)]
            other_examples = [ex for ex in examples if not self._has_challenging_entity(ex)]
            
            # Create smaller batches for challenging examples
            batch_size = 16
            for i in range(0, len(challenging_examples), batch_size):
                batches.append(challenging_examples[i:i + batch_size])
            
            # Create regular batches for other examples
            batch_size = 32
            for i in range(0, len(other_examples), batch_size):
                batches.append(other_examples[i:i + batch_size])
            
            random.shuffle(batches)
            return batches
        else:
            # Use regular batching for later epochs
            batch_size = 32
            return [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]
    
    def _has_challenging_entity(self, example: Example) -> bool:
        """Check if an example contains challenging entities"""
        challenging_entities = {"MATERIAL", "LUMBAR_SUPPORT"}
        return any(ent.label_ in challenging_entities for ent in example.reference.ents)
    
    def _get_dropout_rate(self, batch: List[Example], epoch: int) -> float:
        """Get dynamic dropout rate based on entities in batch"""
        if any(self._has_challenging_entity(ex) for ex in batch):
            return max(0.3 - (epoch / 200), 0.2)  # Higher dropout for challenging entities
        return 0.2  # Default dropout
    
    def _evaluate(self, examples: List[Example]) -> Dict:
        """Evaluate model performance with detailed entity-level metrics"""
        scores = {}
        for example in examples:
            pred_doc = self.nlp(example.reference.text)
            gold_ents = {(ent.label_, ent.start_char, ent.end_char) for ent in example.reference.ents}
            pred_ents = {(ent.label_, ent.start_char, ent.end_char) for ent in pred_doc.ents}
            
            for ent_type in self.nlp.pipe_labels['ner']:
                if ent_type not in scores:
                    scores[ent_type] = {"tp": 0, "fp": 0, "fn": 0}
                
                gold = {(start, end) for (label, start, end) in gold_ents if label == ent_type}
                pred = {(start, end) for (label, start, end) in pred_ents if label == ent_type}
                
                scores[ent_type]["tp"] += len(gold & pred)
                scores[ent_type]["fp"] += len(pred - gold)
                scores[ent_type]["fn"] += len(gold - pred)
        
        # Calculate metrics
        metrics = {
            "entity_level": {"tp": 0, "fp": 0, "fn": 0, "precision": 0, "recall": 0, "f1": 0},
            "per_entity": {}
        }
        
        for ent_type, scores in scores.items():
            tp = scores["tp"]
            fp = scores["fp"]
            fn = scores["fn"]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics["per_entity"][ent_type] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": tp + fn
            }
            
            metrics["entity_level"]["tp"] += tp
            metrics["entity_level"]["fp"] += fp
            metrics["entity_level"]["fn"] += fn
        
        # Calculate overall metrics
        total_tp = metrics["entity_level"]["tp"]
        total_fp = metrics["entity_level"]["fp"]
        total_fn = metrics["entity_level"]["fn"]
        
        metrics["entity_level"]["precision"] = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        metrics["entity_level"]["recall"] = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        metrics["entity_level"]["f1"] = 2 * metrics["entity_level"]["precision"] * metrics["entity_level"]["recall"] / (
            metrics["entity_level"]["precision"] + metrics["entity_level"]["recall"]
        ) if (metrics["entity_level"]["precision"] + metrics["entity_level"]["recall"]) > 0 else 0
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _log_progress(self, epoch: int, losses: Dict, metrics: Dict):
        """Log training progress"""
        print(f"\nEpoch {epoch}")
        print(f"Loss: {losses.get('ner', 0):.4f}")
        print(f"Overall F1: {metrics['entity_level']['f1']:.4f}")
        
        print("\nPer-entity metrics:")
        for entity, scores in metrics['per_entity'].items():
            print(f"{entity:15} F1: {scores['f1']:.4f} (P: {scores['precision']:.4f}, R: {scores['recall']:.4f})")

def train_improved_model(
    input_path: str,
    output_dir: str,
    n_iter: int = 100
) -> Tuple[spacy.language.Language, str]:
    """Main function to train the improved NER model"""
    
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training data
    with open(input_path, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    # Initialize entity balancer with synonyms
    synonyms = {
        "MATERIAL": [
            "leather", "fabric", "cloth", "velvet", "vinyl", "textile",
            "mesh", "polyester", "cotton", "microfiber", "suede", "wool",
            "synthetic", "upholstery", "material", "faux leather"
        ],
        "LUMBAR_SUPPORT": [
            "lumbar support", "lumbar cushion", "back support",
            "lumbar pad", "ergonomic support", "spine support",
            "lower back support", "adjustable lumbar", "lumbar region support"
        ]
    }
    
    balancer = EntityBalancer(synonyms)
    entity_stats = balancer.analyze_entity_distribution(training_data)
    
    # Generate balanced training data
    balanced_data = []
    for entity, stats in entity_stats.items():
        if entity in balancer.weak_entities:
            print(f"Generating additional examples for weak entity: {entity}")
            examples = balancer.generate_balanced_examples(entity, count=200)
            balanced_data.extend(examples)
    
    # Combine with original data
    final_training_data = training_data + balanced_data
    
    # Train model
    trainer = ImprovedNERTrainer(final_training_data, balancer)
    model = trainer.train(n_iter=n_iter)
    
    # Save model
    output_path = os.path.join(output_dir, "improved_ner_model")
    model.to_disk(output_path)
    
    # Create performance report
    report_path = os.path.join(output_dir, "training_report.md")
    create_performance_report(
        report_path,
        trainer.metrics_history,
        time.time() - start_time,
        len(training_data),
        len(balanced_data)
    )
    
    return model, report_path

def create_performance_report(
    report_path: str,
    metrics_history: List[Dict],
    training_time: float,
    original_count: int,
    synthetic_count: int
):
    """Create a detailed performance report"""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# NER Training Performance Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Training Data Composition\n")
        f.write(f"- Original examples: {original_count}\n")
        f.write(f"- Synthetic examples: {synthetic_count}\n")
        f.write(f"- Total examples: {original_count + synthetic_count}\n\n")
        
        f.write("## Training Statistics\n")
        f.write(f"- Training time: {training_time:.2f} seconds\n")
        f.write(f"- Final overall F1: {metrics_history[-1]['entity_level']['f1']:.4f}\n\n")
        
        f.write("## Per-Entity Final Metrics\n")
        f.write("| Entity | F1 Score | Precision | Recall | Support |\n")
        f.write("|--------|----------|-----------|---------|----------|\n")
        
        final_metrics = metrics_history[-1]['per_entity']
        for entity, scores in sorted(final_metrics.items()):
            f.write(f"| {entity} | {scores['f1']:.4f} | {scores['precision']:.4f} | ")
            f.write(f"{scores['recall']:.4f} | {scores['support']} |\n")

if __name__ == "__main__":
    model, report_path = train_improved_model(
        input_path="seat_entities_new_min.json",
        output_dir="improved_ner_output",
        n_iter=100
    )
