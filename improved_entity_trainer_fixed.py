"""
Enhanced NER trainer with improved entity-specific handling and balanced data generation.
Focuses on improving performance for challenging entities like MATERIAL and LUMBAR_SUPPORT.
"""

import spacy
import json
import os
import random
import time
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
from datetime import datetime
import numpy as np
from spacy.training import Example
from visualization_components import (
    ComprehensiveTrainingVisualizer,
    WordCloudGenerator,
    LDAAnalyzer
)

# Fixed random seed for reproducibility
RANDOM_SEED = 42

def process_annotations(item: Dict) -> List[Tuple[int, int, str]]:
    """Process annotations from different formats and return standardized entities"""
    entities = []
    label_map = {
        "recliner": "RECLINER", "seat_message": "SEAT_MESSAGE", "seat_warmer": "SEAT_WARMER",
        "headrest": "HEADREST", "armrest": "ARMREST", "footrest": "FOOTREST",
        "backrest": "BACKREST", "cushion": "CUSHION", "material": "MATERIAL",
        "traytable": "TRAYTABLE", "lumbar_support": "LUMBAR_SUPPORT",
        "seat_material": "MATERIAL", "lumbar": "LUMBAR_SUPPORT"
    }
    
    # Handle label format
    if 'label' in item and isinstance(item['label'], list):
        for label_item in item['label']:
            if isinstance(label_item, dict) and 'labels' in label_item and label_item['labels']:
                start = label_item.get('start')
                end = label_item.get('end')
                if start is not None and end is not None:
                    raw_label = label_item['labels'][0].lower()
                    label = label_map.get(raw_label, raw_label.upper())
                    entities.append((int(start), int(end), label))
    
    # Handle annotations format
    elif 'annotations' in item and item['annotations'] and 'result' in item['annotations'][0]:
        for annotation in item['annotations'][0]['result']:
            value = annotation.get("value", {})
            start = value.get("start")
            end = value.get("end")
            labels = value.get("labels", [])
            if start is not None and end is not None and labels:
                raw_label = labels[0].lower()
                label = label_map.get(raw_label, raw_label.upper())
                entities.append((int(start), int(end), label))
    
    return entities

def load_training_data(file_path: str) -> List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]]:
    """Load and preprocess training data from JSON"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    training_data = []
    for item in raw_data:
        # Extract text
        text = None
        if 'ReviewText' in item:
            text = item['ReviewText']
        elif 'data' in item:
            text_data = item['data']
            text = text_data.get('ReviewText') or text_data.get('Review Text') or text_data.get('feedback')
        
        if not text or not isinstance(text, str):
            continue
            
        text = text.strip()
        if not text:
            continue
        
        # Process annotations
        entities = process_annotations(item)
        if entities:
            # Filter out invalid entity positions
            valid_entities = [
                (start, end, label) for start, end, label in entities
                if 0 <= start < end <= len(text)
            ]
            if valid_entities:
                training_data.append((text, {"entities": valid_entities}))
    
    return training_data

class EntityBalancer:
    """Handles entity-specific data balancing and augmentation strategies"""
    
    def __init__(self, base_synonyms: Dict[str, List[str]]):
        self.base_synonyms = base_synonyms
        self.entity_stats = {}
        self.weak_entities = set()
        
    def analyze_entity_distribution(self, training_data: List[Tuple]) -> Dict:
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
                print(f"Identified weak entity: {entity} (count: {count}, {count/total_entities*100:.1f}%)")
        
        return self.entity_stats
    
    def generate_balanced_examples(self, entity: str, count: int) -> List[Tuple]:
        """Generate balanced examples for a specific entity"""
        examples = []
        synonyms = self.base_synonyms.get(entity, [])
        
        if not synonyms:
            print(f"Warning: No synonyms found for entity {entity}")
            return examples
            
        if entity in self.weak_entities:
            # Generate more diverse examples for weak entities
            count = int(count * 1.5)  # 50% more examples for weak entities
            print(f"Increasing example count for weak entity {entity} to {count}")
            
        templates = self._get_entity_templates(entity)
        for _ in range(count):
            example = self._create_example(entity, templates, synonyms)
            if example:
                examples.append(example)
        
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
    
    def _create_example(self, entity: str, templates: List[str], synonyms: List[str]) -> Optional[Tuple]:
        """Create a single synthetic example"""
        try:
            template = random.choice(templates)
            synonym = random.choice(synonyms)
            text = template.format(value=synonym)
            
            # Find the position of the entity in the text
            start = text.find(synonym)
            if start == -1:  # Shouldn't happen, but let's be safe
                return None
                
            end = start + len(synonym)
            return (text, {"entities": [(start, end, entity)]})
            
        except Exception as e:
            print(f"Error creating example for {entity}: {str(e)}")
            return None

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
            try:
                doc = self.nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                
                entities_in_example = {ent.label_ for ent in example.reference.ents}
                for entity in entities_in_example:
                    entity_examples[entity].append(example)
                all_examples.append(example)
            except Exception as e:
                print(f"Error creating example: {str(e)}")
                continue
        
        train_examples = []
        val_examples = []
        
        # Stratified split for each entity
        for entity, examples in entity_examples.items():
            if not examples:
                continue
                
            random.shuffle(examples)
            val_size = max(min_examples_per_entity, int(len(examples) * 0.2))
            val_examples.extend(examples[:val_size])
            train_examples.extend(examples[val_size:])
        
        # Remove duplicates while preserving order
        train_examples = list(dict.fromkeys(train_examples))
        val_examples = list(dict.fromkeys(val_examples))
        
        print(f"Split: {len(train_examples)} train, {len(val_examples)} validation examples")
        return train_examples, val_examples

    def train(self, n_iter: int = 100) -> spacy.language.Language:
        """Train the model with enhanced entity handling"""
        spacy.util.fix_random_seed(RANDOM_SEED)
        
        # Setup NER
        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe("ner")
        
        # Add labels
        for _, annotations in self.train_data:
            for _, _, label in annotations.get("entities", []):
                if label not in ner.labels:
                    ner.add_label(label)
        
        # Create train/val split
        train_examples, val_examples = self.create_stratified_split()
        if not train_examples:
            raise ValueError("No valid training examples found")
        
        # Initialize the model
        optimizer = self.nlp.initialize(lambda: train_examples)
        
        # Training loop
        patience = 0
        max_patience = 15
        best_f1 = 0.0
        batch_sizes = [4, 8, 16, 32]  # Progressive batch sizes
        
        print("Starting training...")
        for epoch in range(n_iter):
            # Progressive training schedule
            batch_size = batch_sizes[min(epoch // 25, len(batch_sizes)-1)]
            
            # Shuffle with fixed seed for reproducibility
            random.shuffle(train_examples)
            
            # Training step
            losses = {}
            batches = spacy.util.minibatch(train_examples, size=batch_size)
            for batch in batches:
                try:
                    losses.update(self.nlp.update(
                        batch,
                        drop=self._get_dropout_rate(batch, epoch),
                        losses=losses
                    ))
                except Exception as e:
                    print(f"Error during batch update: {str(e)}")
                    continue
            
            # Evaluation
            if val_examples and (epoch + 1) % 5 == 0:  # Evaluate every 5 epochs
                metrics = self._evaluate(val_examples)
                current_f1 = metrics['entity_level']['f1']
                
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    self.best_model_state = self.nlp.to_bytes()
                    patience = 0
                    print(f"New best F1: {best_f1:.4f}")
                else:
                    patience += 1
                
                if patience >= max_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{n_iter}, Loss: {losses.get('ner', 0):.4f}")
        
        # Restore best model
        if self.best_model_state:
            self.nlp.from_bytes(self.best_model_state)
            print(f"Restored best model (F1: {best_f1:.4f})")
        
        return self.nlp
    
    def _get_dropout_rate(self, batch: List[Example], epoch: int) -> float:
        """Get dynamic dropout rate based on entities in batch and training phase"""
        has_challenging = False
        for example in batch:
            if any(ent.label_ in {"MATERIAL", "LUMBAR_SUPPORT"} for ent in example.reference.ents):
                has_challenging = True
                break
        
        if has_challenging:
            base_rate = 0.3
        else:
            base_rate = 0.2
            
        # Decrease dropout gradually
        return max(base_rate - (epoch / 200), 0.1)
    
    def _evaluate(self, examples: List[Example]) -> Dict:
        """Evaluate model performance with detailed entity-level metrics"""
        scores = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        
        for example in examples:
            pred_doc = self.nlp(example.reference.text)
            gold = {(ent.label_, ent.start_char, ent.end_char) for ent in example.reference.ents}
            pred = {(ent.label_, ent.start_char, ent.end_char) for ent in pred_doc.ents}
            
            # Update counts for each entity type
            for ent_type in set(x[0] for x in gold | pred):
                gold_ents = {(s,e) for (l,s,e) in gold if l == ent_type}
                pred_ents = {(s,e) for (l,s,e) in pred if l == ent_type}
                
                scores[ent_type]["tp"] += len(gold_ents & pred_ents)
                scores[ent_type]["fp"] += len(pred_ents - gold_ents)
                scores[ent_type]["fn"] += len(gold_ents - pred_ents)
        
        # Calculate metrics
        metrics = {
            "entity_level": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "per_entity": {}
        }
        
        total_tp = total_fp = total_fn = 0
        
        for ent_type, counts in scores.items():
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics["per_entity"][ent_type] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": tp + fn
            }
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        # Calculate micro-averaged metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics["entity_level"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        self.metrics_history.append(metrics)
        return metrics

def train_improved_model(input_path: str, output_dir: str, n_iter: int = 100) -> Tuple[spacy.language.Language, str]:
    """Main function to train the improved NER model"""
    print(f"Starting improved NER training with {n_iter} iterations")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    
    # Initialize entity balancer with extended synonyms
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
    
    try:
        # Load and preprocess training data
        print("Loading training data...")
        training_data = load_training_data(input_path)
        if not training_data:
            raise ValueError("No valid training data loaded")
            
        print(f"Loaded {len(training_data)} examples")
        
        # Initialize balancer and analyze distribution
        balancer = EntityBalancer(synonyms)
        entity_stats = balancer.analyze_entity_distribution(training_data)
        
        # Generate balanced examples for weak entities
        print("\nGenerating balanced training data...")
        balanced_data = []
        for entity in balancer.weak_entities:
            print(f"Generating examples for {entity}...")
            examples = balancer.generate_balanced_examples(entity, count=200)
            balanced_data.extend(examples)
            print(f"Generated {len(examples)} examples for {entity}")
        
        # Combine original and synthetic data
        final_training_data = training_data + balanced_data
        print(f"\nFinal training data size: {len(final_training_data)}")
        
        # Train model
        print("\nInitializing trainer...")
        trainer = ImprovedNERTrainer(final_training_data, balancer)
        
        print("Starting training...")
        model = trainer.train(n_iter=n_iter)
        
        if not model:
            raise ValueError("Model training failed")
        
        # Save model
        output_path = os.path.join(output_dir, "improved_ner_model")
        model.to_disk(output_path)
        print(f"\nModel saved to: {output_path}")
        
        # Initialize visualizers
        print("\nGenerating visualizations and analytics...")
        visualizer = ComprehensiveTrainingVisualizer(output_dir)
        wordcloud_gen = WordCloudGenerator(output_dir)
        lda_analyzer = LDAAnalyzer(output_dir)
        
        # Extract texts for analysis
        training_texts = [text for text, _ in final_training_data]
        
        # Generate visualizations
        visualizer.plot_training_progress(trainer.metrics_history)
        visualizer.plot_per_entity_progress(trainer.metrics_history)
        visualizer.plot_individual_entity_progress(trainer.metrics_history)
        visualizer.plot_final_results(trainer.metrics_history[-1])
        visualizer.plot_data_composition(len(training_data), len(balanced_data))
        
        # Generate word clouds
        print("Generating word clouds...")
        wordcloud_gen.generate_comprehensive_wordclouds(training_texts)
        
        # Perform LDA analysis
        print("Performing LDA topic modeling...")
        lda_model, dictionary, corpus, coherence = lda_analyzer.perform_lda_analysis(training_texts)
        
        # Create comprehensive report
        report_path = os.path.join(output_dir, "training_report.md")
        create_performance_report(
            report_path,
            trainer.metrics_history,
            time.time() - start_time,
            len(training_data),
            len(balanced_data)
        )
        print(f"Training report saved to: {report_path}")
        
        return model, report_path
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

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
        
        # Data composition
        f.write("## Training Data Composition\n")
        f.write(f"- Original examples: {original_count}\n")
        f.write(f"- Synthetic examples: {synthetic_count}\n")
        f.write(f"- Total examples: {original_count + synthetic_count}\n\n")
        
        # Training statistics
        f.write("## Training Statistics\n")
        f.write(f"- Training time: {training_time:.2f} seconds\n")
        
        if metrics_history:
            final_metrics = metrics_history[-1]
            f.write(f"- Final overall F1: {final_metrics['entity_level']['f1']:.4f}\n")
            f.write(f"- Final precision: {final_metrics['entity_level']['precision']:.4f}\n")
            f.write(f"- Final recall: {final_metrics['entity_level']['recall']:.4f}\n\n")
            
            # Per-entity final metrics
            f.write("## Per-Entity Final Metrics\n")
            f.write("| Entity | F1 Score | Precision | Recall | Support |\n")
            f.write("|--------|----------|-----------|---------|----------|\n")
            
            # Sort entities by F1 score for better readability
            per_entity = final_metrics['per_entity']
            sorted_entities = sorted(
                per_entity.items(),
                key=lambda x: (-x[1]['f1'], x[0])  # Sort by F1 (desc) then name
            )
            
            for entity, scores in sorted_entities:
                f.write(
                    f"| {entity} | {scores['f1']:.4f} | {scores['precision']:.4f} | "
                    f"{scores['recall']:.4f} | {scores['support']} |\n"
                )

if __name__ == "__main__":
    model, report_path = train_improved_model(
        input_path="seat_entities_new_min.json",
        output_dir="improved_ner_output",
        n_iter=100
    )
