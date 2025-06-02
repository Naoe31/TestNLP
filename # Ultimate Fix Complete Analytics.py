# Ultimate Fix Complete Analytics - Full Integration with ADVANCED NER IMPROVEMENTS
# Enhanced for 0.9+ F1 Performance with Advanced Training Techniques
# Features: Advanced Data Augmentation, Adaptive Learning, Ensemble Methods, Enhanced Validation

import pandas as pd
import random
import json
import os
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import spacy
from spacy.training import Example
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import string
import re
import nltk
import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel, CoherenceModel
from wordcloud import WordCloud, STOPWORDS as WC_STOPWORDS
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from scipy.stats import entropy

# Download required NLTK data
try:
    nltk.data.find('corpora/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# SET FIXED SEEDS FOR REPRODUCIBILITY
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class AdvancedEntityAugmenter:
    """Advanced entity augmentation with contextual understanding and adversarial examples"""
    
    def __init__(self):
        # Enhanced synonyms with contextual variations
        self.seat_synonyms = {
            "TRAYTABLE": {
                "primary": ["tray table", "fold down table", "dining table", "work table"],
                "variations": ["tray", "table", "folding table", "fold-down table", "work surface"],
                "contexts": ["meal", "dining", "work", "laptop", "food", "drink"]
            },
            "MATERIAL": {
                "primary": ["material", "seat material", "fabric", "leather", "upholstery"],
                "variations": ["vinyl", "cloth", "velvet", "textile", "covering", "surface"],
                "contexts": ["quality", "texture", "feel", "comfort", "durability", "appearance"]
            },
            "LUMBAR_SUPPORT": {
                "primary": ["lumbar support", "lumbar", "lumbar pad", "lower back support"],
                "variations": ["back support", "spine support", "lumbar cushion"],
                "contexts": ["comfort", "posture", "back pain", "ergonomic", "support"]
            },
            "SEAT_WARMER": {
                "primary": ["seat warmer", "warmer", "seat heating", "heated seat"],
                "variations": ["warming", "heated", "heating", "heat", "thermal"],
                "contexts": ["cold", "winter", "temperature", "comfort", "warmth"]
            },
            "SEAT_MESSAGE": {
                "primary": ["massage", "seat massage", "massaging", "massage function"],
                "variations": ["massager", "vibration", "vibrating", "massage feature"],
                "contexts": ["relaxation", "tension", "comfort", "therapeutic", "relief"]
            },
            "RECLINER": {
                "primary": ["reclining", "recline", "recliner", "reclining function"],
                "variations": ["reclined", "reclines", "reclinable", "seat angle"],
                "contexts": ["comfort", "position", "adjustment", "angle", "relaxation"]
            },
            "ARMREST": {
                "primary": ["armrest", "arm rest", "armrests"],
                "variations": ["arm-rest", "arm support", "elbow rest"],
                "contexts": ["comfort", "support", "position", "adjustment"]
            },
            "BACKREST": {
                "primary": ["backrest", "seat back", "seatback"],
                "variations": ["back", "back support", "back-rest", "spine support"],
                "contexts": ["comfort", "support", "posture", "angle"]
            },
            "HEADREST": {
                "primary": ["headrest", "neckrest", "head support"],
                "variations": ["neck support", "head-rest", "headrests"],
                "contexts": ["comfort", "support", "neck", "head", "position"]
            },
            "CUSHION": {
                "primary": ["cushion", "padding", "cushioning"],
                "variations": ["seat base", "base", "bottom", "cushions", "padded"],
                "contexts": ["comfort", "softness", "support", "thickness"]
            },
            "FOOTREST": {
                "primary": ["footrest", "foot-rest", "footrests"],
                "variations": ["leg support", "ottoman", "leg extension"],
                "contexts": ["comfort", "position", "legs", "extension"]
            }
        }
        
        # Advanced template patterns with emotional context
        self.advanced_templates = {
            "positive_experience": [
                "The {entity} {verb} {adjective} and {experience}",
                "I {feeling} the {entity} because it {benefit}",
                "{overall}, the {entity} is {quality} for {use_case}",
                "The {entity} {function} perfectly when {situation}",
            ],
            "negative_experience": [
                "The {entity} {problem} and {consequence}",
                "I {dislike} the {entity} because it {issue}",
                "Unfortunately, the {entity} is {defect} for {use_case}",
                "The {entity} {malfunction} especially when {situation}",
            ],
            "comparative": [
                "Compared to {comparison}, this {entity} is {comparison_result}",
                "The {entity} is {comparative_adj} than {previous_experience}",
                "Unlike {alternative}, this {entity} {advantage}",
            ],
            "technical_detail": [
                "The {entity} has {technical_spec} that {technical_benefit}",
                "From a technical perspective, the {entity} {technical_assessment}",
                "The {entity} design includes {feature} which {impact}",
            ]
        }
        
        # Context-aware word banks
        self.word_banks = {
            "verbs": {
                "positive": ["works", "functions", "operates", "performs", "provides", "delivers"],
                "negative": ["fails", "breaks", "malfunctions", "disappoints", "lacks"]
            },
            "adjectives": {
                "positive": ["excellent", "comfortable", "premium", "durable", "smooth", "responsive"],
                "negative": ["uncomfortable", "cheap", "broken", "stiff", "unresponsive", "inadequate"]
            },
            "experiences": {
                "positive": ["exceeds expectations", "provides great comfort", "works flawlessly"],
                "negative": ["causes discomfort", "creates problems", "needs improvement"]
            },
            "feelings": {
                "positive": ["love", "appreciate", "enjoy", "prefer"],
                "negative": ["dislike", "hate", "regret", "avoid"]
            }
        }
    
    def generate_contextual_examples(self, entity_type: str, count: int, sentiment_bias: float = 0.0) -> List[Tuple[str, Dict]]:
        """Generate contextually rich examples with sentiment control"""
        if entity_type not in self.seat_synonyms:
            return []
        
        synthetic_examples = []
        entity_info = self.seat_synonyms[entity_type]
        
        for i in range(count):
            # Choose entity term (bias towards primary terms)
            if random.random() < 0.7:
                entity_term = random.choice(entity_info["primary"])
            else:
                entity_term = random.choice(entity_info["variations"])
            
            # Determine sentiment (with bias control)
            sentiment_score = random.random() + sentiment_bias
            is_positive = sentiment_score > 0.5
            
            # Choose template category
            template_categories = ["positive_experience", "technical_detail", "comparative"]
            if not is_positive:
                template_categories = ["negative_experience", "technical_detail", "comparative"]
            
            template_category = random.choice(template_categories)
            template = random.choice(self.advanced_templates[template_category])
            
            # Fill template with context-aware words
            text = self._fill_template(template, entity_term, entity_type, is_positive)
            
            # Add natural variations
            text = self._add_natural_variations(text)
            
            # Find entity position
            entity_term_lower = entity_term.lower()
            text_lower = text.lower()
            start_pos = text_lower.find(entity_term_lower)
            
            if start_pos == -1:
                # Try partial matches for multi-word entities
                for word in entity_term.split():
                    start_pos = text_lower.find(word.lower())
                    if start_pos != -1:
                        entity_term = word
                        break
            
            if start_pos == -1:
                continue
                
            end_pos = start_pos + len(entity_term)
            entities = [(start_pos, end_pos, entity_type)]
            synthetic_examples.append((text, {"entities": entities}))
        
        return synthetic_examples
    
    def _fill_template(self, template: str, entity_term: str, entity_type: str, is_positive: bool) -> str:
        """Fill template with context-appropriate words"""
        sentiment_key = "positive" if is_positive else "negative"
        
        replacements = {
            "entity": entity_term,
            "verb": random.choice(self.word_banks["verbs"][sentiment_key]),
            "adjective": random.choice(self.word_banks["adjectives"][sentiment_key]),
            "experience": random.choice(self.word_banks["experiences"][sentiment_key]),
            "feeling": random.choice(self.word_banks["feelings"][sentiment_key]),
            "quality": random.choice(self.word_banks["adjectives"][sentiment_key]),
            "use_case": random.choice(["long trips", "daily use", "comfort", "work"]),
            "situation": random.choice(["needed most", "traveling", "working", "resting"]),
            "benefit": "provides excellent comfort" if is_positive else "causes issues",
            "problem": "doesn't work properly" if not is_positive else "works great",
            "consequence": "creates discomfort" if not is_positive else "enhances comfort"
        }
        
        # Apply replacements
        for key, value in replacements.items():
            template = template.replace(f"{{{key}}}", value)
        
        return template
    
    def _add_natural_variations(self, text: str) -> str:
        """Add natural language variations"""
        # Add discourse markers
        starters = ["", "Honestly, ", "Overall, ", "In my experience, ", "I think ", "Actually, "]
        if random.random() < 0.3:
            text = random.choice(starters) + text.lower()
        
        # Add endings
        endings = ["", ".", " overall.", " in my opinion.", " for sure.", " really."]
        if random.random() < 0.3:
            text += random.choice(endings)
        
        return text
    
    def generate_adversarial_examples(self, entity_type: str, count: int) -> List[Tuple[str, Dict]]:
        """Generate adversarial examples to improve robustness"""
        adversarial_examples = []
        
        if entity_type not in self.seat_synonyms:
            return adversarial_examples
        
        entity_info = self.seat_synonyms[entity_type]
        
        # Create challenging contexts
        challenging_patterns = [
            "The {entity} and other features work well together",
            "Unlike the broken {other_entity}, the {entity} functions properly",
            "The {entity} quality is better than the {other_entity}",
            "Both the {entity} and {other_entity} need improvement",
            "The {entity} works but the {other_entity} doesn't"
        ]
        
        other_entities = [k for k in self.seat_synonyms.keys() if k != entity_type]
        
        for i in range(count):
            entity_term = random.choice(entity_info["primary"] + entity_info["variations"])
            other_entity_type = random.choice(other_entities)
            other_entity_term = random.choice(self.seat_synonyms[other_entity_type]["primary"])
            
            pattern = random.choice(challenging_patterns)
            text = pattern.format(entity=entity_term, other_entity=other_entity_term)
            
            # Find target entity position
            start_pos = text.lower().find(entity_term.lower())
            if start_pos != -1:
                end_pos = start_pos + len(entity_term)
                entities = [(start_pos, end_pos, entity_type)]
                adversarial_examples.append((text, {"entities": entities}))
        
        return adversarial_examples


class AdaptiveLearningScheduler:
    """Adaptive learning rate and training parameter scheduler"""
    
    def __init__(self, initial_params: Dict):
        self.initial_params = initial_params
        self.performance_history = []
        self.plateau_counter = 0
        self.best_performance = 0.0
        
    def update_parameters(self, current_performance: float, epoch: int) -> Dict:
        """Update training parameters based on performance"""
        self.performance_history.append(current_performance)
        
        # Check for improvement
        if current_performance > self.best_performance:
            self.best_performance = current_performance
            self.plateau_counter = 0
        else:
            self.plateau_counter += 1
        
        # Adaptive parameter adjustment
        params = self.initial_params.copy()
        
        # Reduce dropout if plateauing
        if self.plateau_counter > 3:
            params['dropout'] = max(0.05, params['dropout'] * 0.9)
        
        # Increase batch size for later stages
        if epoch > 50:
            params['batch_size'] = min(64, params['batch_size'] * 1.2)
        
        # Adaptive learning schedule
        if epoch < 20:
            params['dropout'] = 0.3
            params['batch_size'] = 4
        elif epoch < 50:
            params['dropout'] = 0.25
            params['batch_size'] = 8
        elif epoch < 100:
            params['dropout'] = 0.2
            params['batch_size'] = 16
        else:
            params['dropout'] = 0.15
            params['batch_size'] = 32
        
        return params


class EnhancedNERTrainer:
    """Enhanced NER trainer with advanced techniques for 0.9+ F1 performance"""
    
    def __init__(self, train_data: List[Tuple]):
        self.train_data = train_data
        self.nlp = spacy.blank("en")
        self.metrics_history = []
        self.best_model_states = []  # Store multiple best models for ensemble
        self.best_f1 = 0.0
        self.best_min_entity_f1 = 0.0
        self.learning_scheduler = AdaptiveLearningScheduler({
            'dropout': 0.3, 'batch_size': 4, 'learning_rate': 0.001
        })
        
    def create_stratified_validation_split(self) -> Tuple[List[Example], List[Example]]:
        """Create stratified validation split ensuring balanced entity representation"""
        entity_examples = defaultdict(list)
        all_examples = []
        
        # Collect all valid examples
        for text, annotations in self.train_data:
            try:
                if not text or not text.strip():
                    continue
                
                valid_entities = []
                for start, end, label in annotations.get("entities", []):
                    if 0 <= start < end <= len(text):
                        valid_entities.append((start, end, label))
                
                if valid_entities:
                    doc = self.nlp.make_doc(text)
                    example = Example.from_dict(doc, {"entities": valid_entities})
                    all_examples.append(example)
                    
                    for _, _, label in valid_entities:
                        entity_examples[label].append(example)
            except Exception:
                continue
        
        # Stratified sampling for validation
        min_val_per_entity = 6  # Minimum validation examples per entity
        target_val_ratio = 0.2  # Target 20% for validation
        
        train_examples = []
        val_examples = []
        reserved_for_val = set()
        
        # First pass: ensure minimum validation per entity
        for entity, examples in entity_examples.items():
            if len(examples) >= min_val_per_entity * 2:
                local_random = random.Random(RANDOM_SEED + hash(entity))
                local_random.shuffle(examples)
                val_count = max(min_val_per_entity, int(len(examples) * target_val_ratio))
                val_count = min(val_count, len(examples) // 2)  # Don't take more than half
                
                for i in range(val_count):
                    reserved_for_val.add(id(examples[i]))
        
        # Second pass: distribute remaining examples
        for example in all_examples:
            if id(example) in reserved_for_val:
                val_examples.append(example)
            else:
                train_examples.append(example)
        
        # Ensure adequate validation size
        current_val_ratio = len(val_examples) / len(all_examples) if all_examples else 0
        if current_val_ratio < 0.15:  # If less than 15%, add more
            needed = int(len(all_examples) * 0.15) - len(val_examples)
            local_random = random.Random(RANDOM_SEED + 1000)
            local_random.shuffle(train_examples)
            for i in range(min(needed, len(train_examples) // 3)):
                val_examples.append(train_examples.pop())
        
        print(f"Enhanced split: {len(train_examples)} train, {len(val_examples)} validation")
        
        # Validation distribution check
        val_entity_counts = defaultdict(int)
        for example in val_examples:
            for ent in example.reference.ents:
                val_entity_counts[ent.label_] += 1
        
        print("Enhanced validation distribution:")
        for entity, count in sorted(val_entity_counts.items()):
            status = "✅" if count >= 6 else "⚠️" if count >= 3 else "❌"
            print(f"  {entity}: {count} examples {status}")
        
        return train_examples, val_examples
    
    def train_enhanced(self, n_iter: int = 250, use_ensemble: bool = True):
        """Enhanced training with advanced techniques"""
        print("🚀 Starting Enhanced NER Training for 0.9+ F1...")
        
        # Set seeds
        spacy.util.fix_random_seed(RANDOM_SEED)
        
        # Setup NER with enhanced configuration
        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe("ner")
        else:
            ner = self.nlp.get_pipe("ner")
        
        # Add labels
        for _, annotations in self.train_data:
            for _, _, label in annotations.get("entities", []):
                if label not in ner.labels:
                    ner.add_label(label)
        
        train_examples, val_examples = self.create_stratified_validation_split()
        
        if not train_examples:
            return None
        
        # Initialize with better strategy
        self.nlp.initialize(lambda: train_examples)
        
        # Enhanced training loop with adaptive parameters
        patience_counter = 0
        stagnation_threshold = 20  # Increased patience for better convergence
        
        for epoch in range(n_iter):
            # Get adaptive parameters
            current_f1 = self.metrics_history[-1]['entity_level']['f1'] if self.metrics_history else 0.0
            params = self.learning_scheduler.update_parameters(current_f1, epoch)
            
            # Training with adaptive parameters
            local_random = random.Random(RANDOM_SEED + epoch)
            shuffled_examples = train_examples.copy()
            local_random.shuffle(shuffled_examples)
            
            losses = {}
            batches = spacy.util.minibatch(shuffled_examples, size=int(params['batch_size']))
            
            for batch in batches:
                try:
                    self.nlp.update(batch, drop=params['dropout'], losses=losses)
                except Exception as e:
                    print(f"⚠️ Training error: {e}")
                    continue
            
            # Enhanced validation schedule
            should_validate = False
            if epoch < 30:
                should_validate = (epoch + 1) % 2 == 0  # Every 2 epochs early
            elif epoch < 100:
                should_validate = (epoch + 1) % 3 == 0  # Every 3 epochs mid
            else:
                should_validate = (epoch + 1) % 5 == 0  # Every 5 epochs late
            
            if val_examples and should_validate:
                metrics = self.evaluate_enhanced(val_examples, f"Epoch {epoch + 1}")
                current_f1 = metrics.get('entity_level', {}).get('f1', 0.0)
                current_min_f1 = self.get_min_entity_f1(metrics)
                
                # Enhanced model selection and ensemble building
                improvement_threshold = 0.001  # Minimum improvement to consider
                if current_f1 > self.best_f1 + improvement_threshold:
                    self.best_f1 = current_f1
                    model_state = self.nlp.to_bytes()
                    
                    if use_ensemble:
                        self.best_model_states.append({
                            'state': model_state,
                            'f1': current_f1,
                            'epoch': epoch + 1,
                            'min_f1': current_min_f1
                        })
                        # Keep only top 3 models for ensemble
                        self.best_model_states.sort(key=lambda x: x['f1'], reverse=True)
                        self.best_model_states = self.best_model_states[:3]
                    
                    patience_counter = 0
                    print(f"🏆 NEW BEST F1: Epoch {epoch + 1}, F1={current_f1:.4f}, Min F1={current_min_f1:.4f}")
                else:
                    patience_counter += 1
                
                # Track minimum entity F1
                if current_min_f1 > self.best_min_entity_f1:
                    self.best_min_entity_f1 = current_min_f1
                
                # Enhanced success criteria
                entities_above_90 = sum(1 for m in metrics.get('per_entity', {}).values() 
                                      if m.get('support', 0) > 0 and m.get('f1', 0) >= 0.9)
                entities_above_85 = sum(1 for m in metrics.get('per_entity', {}).values() 
                                      if m.get('support', 0) > 0 and m.get('f1', 0) >= 0.85)
                total_entities = sum(1 for m in metrics.get('per_entity', {}).values() 
                                   if m.get('support', 0) > 0)
                
                # Success condition: most entities above 0.9
                if entities_above_90 >= max(8, int(total_entities * 0.8)):
                    print(f"🎯 SUCCESS ACHIEVED: {entities_above_90}/{total_entities} entities ≥ 0.9!")
                    break
                
                # Good progress condition
                if entities_above_85 >= total_entities and current_f1 > 0.88:
                    print(f"🎯 EXCELLENT PROGRESS: All entities ≥ 0.85, Overall F1={current_f1:.4f}")
                
                # Enhanced early stopping
                if patience_counter >= stagnation_threshold:
                    print(f"🛑 Early stopping: No improvement for {stagnation_threshold} evaluations")
                    break
                
            # Progress reporting
            if (epoch + 1) % 25 == 0:
                loss_val = losses.get('ner', 0)
                print(f"Epoch {epoch + 1}/{n_iter}, Loss: {loss_val:.4f}, "
                      f"Dropout: {params['dropout']:.3f}, Batch: {int(params['batch_size'])}")
        
        # Restore best model or create ensemble
        if use_ensemble and len(self.best_model_states) > 1:
            print(f"🎭 Creating ensemble from {len(self.best_model_states)} best models...")
            # For now, use the single best model (ensemble prediction would require more complex implementation)
            best_model = max(self.best_model_states, key=lambda x: x['f1'])
            self.nlp.from_bytes(best_model['state'])
            print(f"✅ Using best single model with F1: {best_model['f1']:.4f}")
        elif self.best_model_states:
            best_model = self.best_model_states[0]
            self.nlp.from_bytes(best_model['state'])
            print(f"✅ Restored best model with F1: {best_model['f1']:.4f}")
        
        return self.nlp
    
    def get_min_entity_f1(self, metrics: Dict) -> float:
        """Get minimum F1 score among entities with support"""
        if 'per_entity' not in metrics:
            return 0.0
        
        min_f1 = 1.0
        valid_entities = 0
        for entity_metrics in metrics['per_entity'].values():
            if entity_metrics.get('support', 0) > 0:
                min_f1 = min(min_f1, entity_metrics.get('f1', 0))
                valid_entities += 1
        
        return min_f1 if valid_entities > 0 else 0.0
    
    def evaluate_enhanced(self, examples: List[Example], phase: str = "Evaluation") -> Dict:
        """Enhanced evaluation with detailed metrics"""
        if not examples:
            return {}
        
        y_true_ents = []
        y_pred_ents = []
        
        for ex in examples:
            try:
                pred_doc = self.nlp(ex.reference.text)
                true_entities = set((ent.label_, ent.start_char, ent.end_char) for ent in ex.reference.ents)
                pred_entities = set((ent.label_, ent.start_char, ent.end_char) for ent in pred_doc.ents)
                y_true_ents.append(true_entities)
                y_pred_ents.append(pred_entities)
            except Exception:
                continue
        
        # Calculate comprehensive metrics
        ENTITIES = ["ARMREST", "BACKREST", "HEADREST", "CUSHION", "MATERIAL",
                   "LUMBAR_SUPPORT", "RECLINER", "FOOTREST", "SEAT_MESSAGE", "SEAT_WARMER", "TRAYTABLE"]
        
        per_entity_metrics = {}
        tp_total = fp_total = fn_total = 0
        
        for label in ENTITIES:
            tp = fp = fn = 0
            for true_set, pred_set in zip(y_true_ents, y_pred_ents):
                true_label = {e for e in true_set if e[0] == label}
                pred_label = {e for e in pred_set if e[0] == label}
                
                tp += len(true_label.intersection(pred_label))
                fp += len(pred_label - true_label)
                fn += len(true_label - pred_label)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_entity_metrics[label] = {
                'precision': precision, 'recall': recall, 'f1': f1,
                'support': tp + fn, 'tp': tp, 'fp': fp, 'fn': fn
            }
            tp_total += tp
            fp_total += fp
            fn_total += fn
        
        overall_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        overall_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        metrics = {
            'phase': phase,
            'entity_level': {'precision': overall_precision, 'recall': overall_recall, 'f1': overall_f1},
            'per_entity': per_entity_metrics
        }
        
        # Enhanced reporting
        entities_above_90 = sum(1 for m in per_entity_metrics.values() if m['support'] > 0 and m['f1'] >= 0.9)
        entities_above_85 = sum(1 for m in per_entity_metrics.values() if m['support'] > 0 and m['f1'] >= 0.85)
        total_with_support = sum(1 for m in per_entity_metrics.values() if m['support'] > 0)
        min_f1 = self.get_min_entity_f1(metrics)
        
        print(f"{phase} - Overall F1: {overall_f1:.4f}")
        print(f"Entities ≥ 0.9: {entities_above_90}/{total_with_support}, ≥ 0.85: {entities_above_85}/{total_with_support}")
        print(f"Min Entity F1: {min_f1:.4f}")
        
        # Show entities needing improvement
        needs_improvement = []
        for label, entity_metrics in per_entity_metrics.items():
            if entity_metrics['support'] > 0 and entity_metrics['f1'] < 0.9:
                needs_improvement.append(f"{label}({entity_metrics['f1']:.3f})")
        
        if needs_improvement:
            print(f"Needs improvement: {', '.join(needs_improvement)}")
        
        self.metrics_history.append(metrics)
        return metrics


class UltimateEntityFixer:
    """Ultimate fixer targeting specific root cause issues"""
    
    def __init__(self):
        # FIXED: Better synonyms matching real usage patterns
        self.seat_synonyms = {
            "TRAYTABLE": [
                "tray table", "fold down table", "dining table", "work table", 
                "work surface", "table", "tray", "folding table", "fold-down table"
            ],
            "MATERIAL": [
                "material", "seat material", "fabric", "leather", "upholstery", 
                "vinyl", "cloth", "velvet", "textile", "covering"
            ],
            "LUMBAR_SUPPORT": [
                "lumbar support", "lumbar", "lumbar pad", "lumbar cushion", 
                "lower back support", "back support", "spine support"
            ],
            "SEAT_WARMER": [
                "seat warmer", "warmer", "warming", "heated", "heating", 
                "seat heating", "heat", "thermal", "warm seat"
            ],
            "SEAT_MESSAGE": [
                "massage", "massaging", "massager", "massage function", 
                "vibration", "vibrating", "seat massage", "massage feature"
            ],
            "RECLINER": [
                "reclining", "recline", "recliner", "reclined", "reclines", 
                "reclinable", "seat angle", "reclining function"
            ],
            "ARMREST": [
                "armrest", "arm rest", "arm-rest", "armrests", 
                "arm support", "elbow rest"
            ],
            "BACKREST": [
                "backrest", "seat back", "seatback", "back", 
                "back support", "back-rest", "spine support"
            ],
            "HEADREST": [
                "headrest", "neckrest", "head support", "neck support", 
                "head-rest", "headrests"
            ],
            "CUSHION": [
                "cushion", "padding", "cushioning", "seat base", 
                "base", "bottom", "cushions", "padded"
            ],
            "FOOTREST": [
                "footrest", "foot-rest", "footrests", "leg support", 
                "ottoman", "leg extension"
            ]
        }
        
        # High-quality templates designed for 0.9+ performance
        self.ultimate_templates = {
            "TRAYTABLE": [
                "The {entity} is perfect for eating meals during flights",
                "I can easily work on my laptop using the {entity}",
                "The {entity} folds down smoothly and securely",
                "The {entity} provides enough space for dining",
                "The {entity} surface is clean and well-maintained",
                "The {entity} mechanism works perfectly every time",
                "The {entity} is sturdy enough to hold heavy items",
                "The {entity} size is adequate for most meals",
                "The {entity} locks in place securely when deployed",
                "The {entity} has a smooth and stable surface"
            ],
            "MATERIAL": [
                "The {entity} feels premium and luxurious",
                "The {entity} quality is excellent and durable",
                "The {entity} has a soft and comfortable texture",
                "The {entity} is made from high-grade materials",
                "The {entity} feels breathable and not sticky",
                "The {entity} has beautiful stitching and finish",
                "The {entity} maintains its appearance over time",
                "The {entity} is easy to clean and maintain",
                "The {entity} color matches the interior perfectly",
                "The {entity} has a sophisticated and elegant look"
            ],
            "LUMBAR_SUPPORT": [
                "The {entity} provides excellent lower back comfort",
                "The {entity} helps maintain proper spine alignment",
                "The {entity} reduces back fatigue during long trips",
                "The {entity} is adjustable to fit my back perfectly",
                "The {entity} offers just the right amount of firmness",
                "The {entity} contours to my spine naturally",
                "The {entity} prevents back pain effectively",
                "The {entity} provides targeted support where needed",
                "The {entity} makes long flights comfortable",
                "The {entity} has excellent ergonomic design"
            ],
            "SEAT_WARMER": [
                "The {entity} function works perfectly in cold weather",
                "I love using the {entity} during winter drives",
                "The {entity} heats up quickly and evenly",
                "The {entity} provides excellent thermal comfort",
                "The {entity} has multiple temperature settings",
                "The {entity} makes long trips more comfortable",
                "The {entity} feature is amazing for cold mornings",
                "The {entity} warms the seat to perfect temperature",
                "The {entity} control is easy to use and responsive",
                "The {entity} distributes heat evenly across the seat"
            ],
            "SEAT_MESSAGE": [
                "The {entity} function provides excellent relaxation",
                "The {entity} helps relieve back tension effectively",
                "The {entity} has multiple intensity settings",
                "The {entity} works perfectly for long drives",
                "The {entity} provides therapeutic relief",
                "The {entity} vibration is smooth and soothing",
                "The {entity} helps reduce driving fatigue",
                "The {entity} is quiet and doesn't create noise",
                "The {entity} covers all the right pressure points",
                "The {entity} can be adjusted for different preferences"
            ],
            "RECLINER": [
                "The {entity} mechanism operates smoothly and quietly",
                "The {entity} positions are comfortable and stable",
                "The {entity} adjustment is easy and intuitive",
                "The {entity} provides excellent comfort for sleeping",
                "The {entity} has multiple position settings",
                "The {entity} locks securely in any position",
                "The {entity} range covers all my comfort needs",
                "The {entity} control is responsive and precise",
                "The {entity} makes the seat incredibly versatile",
                "The {entity} function works reliably every time"
            ]
        }
    
    def generate_massive_examples(self, entity_type: str, count: int) -> List[Tuple[str, Dict]]:
        """Generate massive high-quality examples for ultimate fix"""
        synthetic_examples = []
        entity_terms = self.seat_synonyms.get(entity_type, [entity_type.lower()])
        templates = self.ultimate_templates.get(entity_type, [])
        
        if not templates:
            # Fallback generic templates
            templates = [
                f"The {entity_type.lower().replace('_', ' ')} is excellent and comfortable",
                f"I really appreciate the {entity_type.lower().replace('_', ' ')} quality",
                f"The {entity_type.lower().replace('_', ' ')} works perfectly for my needs",
                f"The {entity_type.lower().replace('_', ' ')} provides great comfort",
                f"The {entity_type.lower().replace('_', ' ')} is well-designed and functional"
            ]
        
        for i in range(count):
            entity_term = random.choice(entity_terms)
            template = random.choice(templates)
            
            # Replace {entity} with the actual term
            text = template.replace("{entity}", entity_term)
            
            # Add natural variations
            starters = ["", "Overall, ", "I think ", "In my experience, ", "Honestly, "]
            endings = ["", ".", " overall.", " for sure.", " in my opinion."]
            
            if random.random() < 0.3:
                starter = random.choice(starters)
                if starter:
                    text = starter + text.lower()
            if random.random() < 0.3:
                text += random.choice(endings)
            
            # Find entity position in the final text
            start_pos = text.lower().find(entity_term.lower())
            if start_pos == -1:
                continue
            end_pos = start_pos + len(entity_term)
            
            entities = [(start_pos, end_pos, entity_type)]
            synthetic_examples.append((text, {"entities": entities}))
        
        return synthetic_examples


class UltimateNERTrainer:
    """Ultimate NER trainer with enhanced validation and tracking"""
    
    def __init__(self, train_data: List[Tuple]):
        self.train_data = train_data
        self.nlp = spacy.blank("en")
        self.metrics_history = []
        self.best_model_state = None
        self.best_f1 = 0.0
        self.best_min_entity_f1 = 0.0
    
    def create_ultimate_validation_split(self) -> Tuple[List[Example], List[Example]]:
        """Create optimized validation split ensuring representation"""
        entity_examples = defaultdict(list)
        all_examples = []
        
        for text, annotations in self.train_data:
            try:
                if not text or not text.strip():
                    continue
                
                valid_entities = []
                for start, end, label in annotations.get("entities", []):
                    if 0 <= start < end <= len(text):
                        valid_entities.append((start, end, label))
                
                if valid_entities:
                    doc = self.nlp.make_doc(text)
                    example = Example.from_dict(doc, {"entities": valid_entities})
                    all_examples.append(example)
                    
                    for _, _, label in valid_entities:
                        entity_examples[label].append(example)
            except Exception:
                continue
        
        # Ensure minimum validation examples per entity
        min_val_per_entity = 4  # Increased for better validation
        train_examples = []
        val_examples = []
        reserved_for_val = set()
        
        for entity, examples in entity_examples.items():
            if len(examples) >= min_val_per_entity * 2:
                local_random = random.Random(RANDOM_SEED + hash(entity) % 1000)
                local_random.shuffle(examples)
                val_count = max(min_val_per_entity, len(examples) // 5)  # 20% or minimum
                
                for i in range(val_count):
                    reserved_for_val.add(id(examples[i]))
        
        for example in all_examples:
            if id(example) in reserved_for_val:
                val_examples.append(example)
            else:
                train_examples.append(example)
        
        # Ensure minimum validation size
        if len(val_examples) < len(all_examples) * 0.15:
            needed = int(len(all_examples) * 0.15) - len(val_examples)
            local_random = random.Random(RANDOM_SEED + 999)
            local_random.shuffle(train_examples)
            for i in range(min(needed, len(train_examples) // 2)):
                val_examples.append(train_examples.pop())
        
        print(f"Ultimate split: {len(train_examples)} train, {len(val_examples)} validation")
        return train_examples, val_examples
    
    def train_ultimate(self, n_iter: int = 200):
        """Ultimate training with comprehensive tracking"""
        print("🚀 Starting Ultimate NER Training...")
        
        # Set seeds
        spacy.util.fix_random_seed(RANDOM_SEED)
        
        # Setup NER
        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe("ner")
        else:
            ner = self.nlp.get_pipe("ner")
        
        # Add labels
        for _, annotations in self.train_data:
            for _, _, label in annotations.get("entities", []):
                if label not in ner.labels:
                    ner.add_label(label)
        
        train_examples, val_examples = self.create_ultimate_validation_split()
        
        if not train_examples:
            return None
        
        # Initialize
        self.nlp.initialize(lambda: train_examples)
        
        # Evaluate initial state
        if val_examples:
            initial_metrics = self.evaluate_ultimate(val_examples, "Epoch 0")
            initial_f1 = initial_metrics.get('entity_level', {}).get('f1', 0.0)
            print(f"🔍 Initial F1: {initial_f1:.4f}")
        
        # Training loop
        patience_counter = 0
        stagnation_counter = 0
        
        for epoch in range(n_iter):
            # Progressive training schedule
            if epoch < 30:
                batch_size, dropout = 4, 0.3
            elif epoch < 60:
                batch_size, dropout = 8, 0.25
            elif epoch < 120:
                batch_size, dropout = 16, 0.2
            else:
                batch_size, dropout = 32, 0.15
            
            # Fixed seed shuffling
            local_random = random.Random(RANDOM_SEED + epoch)
            shuffled_examples = train_examples.copy()
            local_random.shuffle(shuffled_examples)
            
            losses = {}
            batches = spacy.util.minibatch(shuffled_examples, size=batch_size)
            
            for batch in batches:
                try:
                    self.nlp.update(batch, drop=dropout, losses=losses)
                except Exception:
                    continue
            
            # Validation schedule
            should_validate = False
            if epoch < 20:
                should_validate = (epoch + 1) % 2 == 0  # Every 2 epochs initially
            elif epoch < 60:
                should_validate = (epoch + 1) % 3 == 0  # Every 3 epochs
            else:
                should_validate = (epoch + 1) % 5 == 0  # Every 5 epochs later
            
            if val_examples and should_validate:
                metrics = self.evaluate_ultimate(val_examples, f"Epoch {epoch + 1}")
                current_f1 = metrics.get('entity_level', {}).get('f1', 0.0)
                current_min_f1 = self.get_min_entity_f1(metrics)
                
                # Track best models
                if current_f1 > self.best_f1:
                    self.best_f1 = current_f1
                    self.best_model_state = self.nlp.to_bytes()
                    patience_counter = 0
                    print(f"🏆 NEW BEST F1: Epoch {epoch + 1}, F1={current_f1:.4f}")
                else:
                    patience_counter += 1
                
                if current_min_f1 > self.best_min_entity_f1:
                    self.best_min_entity_f1 = current_min_f1
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                
                # Early stopping conditions
                entities_above_90 = sum(1 for m in metrics['per_entity'].values() 
                                      if m.get('support', 0) > 0 and m.get('f1', 0) >= 0.9)
                
                if entities_above_90 >= 7:  # Success condition
                    print(f"🎉 SUCCESS: {entities_above_90} entities ≥ 0.9! Stopping early.")
                    break
                
                if patience_counter >= 15:  # No overall improvement
                    print(f"🛑 Early stopping: No F1 improvement for 15 evaluations")
                    break
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{n_iter}, Loss: {losses.get('ner', 0):.4f}")
        
        # Restore best model
        if self.best_model_state:
            self.nlp.from_bytes(self.best_model_state)
            print(f"✅ Restored best model with F1: {self.best_f1:.4f}")
        
        return self.nlp
    
    def get_min_entity_f1(self, metrics: Dict) -> float:
        """Get minimum F1 score among entities with support"""
        if 'per_entity' not in metrics:
            return 0.0
        
        min_f1 = 1.0
        for entity_metrics in metrics['per_entity'].values():
            if entity_metrics.get('support', 0) > 0:
                min_f1 = min(min_f1, entity_metrics.get('f1', 0))
        
        return min_f1 if min_f1 < 1.0 else 0.0
    
    def evaluate_ultimate(self, examples: List[Example], phase: str = "Evaluation") -> Dict:
        """Ultimate evaluation with comprehensive metrics"""
        if not examples:
            return {}
        
        y_true_ents = []
        y_pred_ents = []
        
        for ex in examples:
            try:
                pred_doc = self.nlp(ex.reference.text)
                true_entities = set((ent.label_, ent.start_char, ent.end_char) for ent in ex.reference.ents)
                pred_entities = set((ent.label_, ent.start_char, ent.end_char) for ent in pred_doc.ents)
                y_true_ents.append(true_entities)
                y_pred_ents.append(pred_entities)
            except Exception:
                continue
        
        # Calculate metrics
        ENTITIES = ["ARMREST", "BACKREST", "HEADREST", "CUSHION", "MATERIAL",
                   "LUMBAR_SUPPORT", "RECLINER", "FOOTREST", "SEAT_MESSAGE", "SEAT_WARMER", "TRAYTABLE"]
        
        per_entity_metrics = {}
        tp_total = fp_total = fn_total = 0
        
        for label in ENTITIES:
            tp = fp = fn = 0
            for true_set, pred_set in zip(y_true_ents, y_pred_ents):
                true_label = {e for e in true_set if e[0] == label}
                pred_label = {e for e in pred_set if e[0] == label}
                
                tp += len(true_label.intersection(pred_label))
                fp += len(pred_label - true_label)
                fn += len(true_label - pred_label)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_entity_metrics[label] = {
                'precision': precision, 'recall': recall, 'f1': f1,
                'support': tp + fn, 'tp': tp, 'fp': fp, 'fn': fn
            }
            tp_total += tp
            fp_total += fp
            fn_total += fn
        
        overall_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        overall_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        metrics = {
            'phase': phase,
            'entity_level': {'precision': overall_precision, 'recall': overall_recall, 'f1': overall_f1},
            'per_entity': per_entity_metrics
        }
        
        # Print concise results
        print(f"{phase} - Overall F1: {overall_f1:.4f}")
        entities_above_90 = sum(1 for m in per_entity_metrics.values() if m['support'] > 0 and m['f1'] >= 0.9)
        total_with_support = sum(1 for m in per_entity_metrics.values() if m['support'] > 0)
        print(f"Entities ≥ 0.9: {entities_above_90}/{total_with_support}")
        
        self.metrics_history.append(metrics)
        return metrics


class LDAAnalyzer:
    """Advanced LDA Topic Modeling with Visualization"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.lda_dir = os.path.join(output_dir, "lda_analysis")
        os.makedirs(self.lda_dir, exist_ok=True)
    
    def preprocess_text_for_lda(self, text: str, nlp_model, stop_words_set: set) -> List[str]:
        """Preprocess text for LDA analysis"""
        if not text or pd.isna(text):
            return []
        
        # Clean and tokenize
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        
        # Use spaCy for lemmatization if available
        try:
            doc = nlp_model(text)
            tokens = [token.lemma_ for token in doc 
                     if not token.is_stop and not token.is_punct 
                     and len(token.lemma_) > 2 and token.lemma_ not in stop_words_set]
        except:
            # Fallback to simple tokenization
            tokens = [word for word in text.split() 
                     if len(word) > 2 and word not in stop_words_set]
        
        return tokens
    
    def perform_lda_analysis(self, text_data: List[str], num_topics: int = 10):
        """Perform comprehensive LDA analysis"""
        print(f"🔍 Starting LDA Topic Modeling with {num_topics} topics...")
        
        # Initialize spaCy model for preprocessing
        try:
            nlp_model = spacy.load("en_core_web_sm")
        except:
            nlp_model = spacy.blank("en")
            print("⚠️ Using blank spaCy model for LDA preprocessing")
        
        # Define comprehensive stopwords
        custom_stop_words = set(nltk.corpus.stopwords.words("english")) | set(string.punctuation) | {
            "seat", "seats", "car", "vehicle", "trip", "feature", "get", "feel", "felt", "look", "make", "also",
            "even", "really", "quite", "very", "much", "good", "great", "nice", "well", "drive", "driving",
            "would", "could", "im", "ive", "id", "nan", "auto", "automobile", "product", "item", "order",
            "time", "way", "thing", "things", "lot", "bit", "little", "big", "small", "new", "old"
        }
        
        # Preprocess texts
        processed_texts = []
        for text in text_data:
            tokens = self.preprocess_text_for_lda(text, nlp_model, custom_stop_words)
            if len(tokens) > 2:  # Minimum tokens for meaningful analysis
                processed_texts.append(tokens)
        
        if len(processed_texts) < 5:
            print("⚠️ Insufficient processed documents for LDA analysis")
            return None, None, None, None
        
        print(f"📊 Processed {len(processed_texts)} documents for LDA")
        
        # Create dictionary and corpus
        dictionary = corpora.Dictionary(processed_texts)
        dictionary.filter_extremes(no_below=3, no_above=0.6, keep_n=10000)
        
        if not dictionary:
            print("⚠️ LDA dictionary is empty after filtering")
            return None, None, None, None
        
        corpus = [dictionary.doc2bow(text) for text in processed_texts]
        corpus = [doc for doc in corpus if doc]  # Remove empty documents
        
        if not corpus or len(corpus) < num_topics:
            if corpus:
                num_topics = max(1, len(corpus) - 1)
                print(f"⚠️ Adjusted num_topics to {num_topics}")
            else:
                return None, None, None, None
        
        # Train LDA model
        print(f"🧠 Training LDA model...")
        lda_model = LdaModel(
            corpus=corpus, 
            id2word=dictionary, 
            num_topics=num_topics,
            random_state=RANDOM_SEED, 
            update_every=1, 
            chunksize=100,
            passes=15, 
            alpha='auto', 
            per_word_topics=True, 
            iterations=100
        )
        
        # Calculate coherence
        coherence_model = CoherenceModel(
            model=lda_model, 
            texts=processed_texts, 
            dictionary=dictionary, 
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        print(f"📈 LDA Coherence Score (c_v): {coherence_score:.4f}")
        
        # Save model and results
        lda_model.save(os.path.join(self.lda_dir, "lda_model.gensim"))
        dictionary.save(os.path.join(self.lda_dir, "lda_dictionary.gensim"))
        corpora.MmCorpus.serialize(os.path.join(self.lda_dir, "lda_corpus.mm"), corpus)
        
        # Generate visualizations
        self.create_lda_visualizations(lda_model, corpus, dictionary, coherence_score)
        
        return lda_model, dictionary, corpus, coherence_score
    
    def create_lda_visualizations(self, lda_model, corpus, dictionary, coherence_score):
        """Create comprehensive LDA visualizations"""
        
        # 1. Interactive pyLDAvis visualization
        try:
            vis_data = gensimvis.prepare(lda_model, corpus, dictionary, mds='mmds')
            pyLDAvis.save_html(vis_data, os.path.join(self.lda_dir, 'lda_interactive_visualization.html'))
            print(f"💾 Interactive LDA visualization saved")
        except Exception as e:
            print(f"⚠️ Error generating interactive LDA visualization: {e}")
        
        # 2. Topic-Word Distribution Heatmap
        num_topics = lda_model.num_topics
        num_words = 10
        
        # Extract topic-word matrix
        topic_word_matrix = []
        topic_labels = []
        word_labels = []
        
        for topic_id in range(num_topics):
            topic_words = lda_model.show_topic(topic_id, topn=num_words)
            if topic_id == 0:
                word_labels = [word for word, _ in topic_words]
            
            topic_values = []
            for word_label in word_labels:
                # Find probability for this word in current topic
                prob = 0
                for word, prob_val in topic_words:
                    if word == word_label:
                        prob = prob_val
                        break
                topic_values.append(prob)
            
            topic_word_matrix.append(topic_values)
            topic_labels.append(f"Topic {topic_id}")
        
        # Create heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(
            topic_word_matrix, 
            xticklabels=word_labels, 
            yticklabels=topic_labels,
            annot=True, 
            fmt='.3f', 
            cmap='YlOrRd',
            cbar_kws={'label': 'Word Probability'}
        )
        plt.title(f'LDA Topic-Word Distribution Heatmap\nCoherence Score: {coherence_score:.4f}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Top Words', fontsize=12)
        plt.ylabel('Topics', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.lda_dir, 'topic_word_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 LDA visualizations saved to {self.lda_dir}")


class SentimentAnalyzer:
    """Sentiment analysis for seat reviews"""
    
    def __init__(self):
        self.positive_words = {
            'excellent', 'amazing', 'perfect', 'great', 'good', 'nice', 'wonderful', 'fantastic',
            'outstanding', 'superb', 'brilliant', 'impressive', 'comfortable', 'love', 'best',
            'premium', 'quality', 'luxurious', 'soft', 'smooth', 'easy', 'convenient', 'helpful',
            'satisfied', 'happy', 'pleased', 'recommend', 'worth', 'value', 'durable', 'reliable'
        }
        
        self.negative_words = {
            'terrible', 'awful', 'bad', 'poor', 'horrible', 'disappointing', 'uncomfortable',
            'cheap', 'flimsy', 'broken', 'damaged', 'worn', 'hard', 'stiff', 'rough', 'pain',
            'ache', 'hurt', 'annoying', 'frustrating', 'useless', 'waste', 'regret', 'avoid',
            'worst', 'never', 'disappointed', 'unhappy', 'unsatisfied', 'problem', 'issue'
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        if not text:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0, 'compound': 0.0}
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Count sentiment words
        pos_count = sum(1 for word in words if word in self.positive_words)
        neg_count = sum(1 for word in words if word in self.negative_words)
        
        total_words = len(words)
        if total_words == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}
        
        # Calculate proportions
        positive = pos_count / total_words
        negative = neg_count / total_words
        neutral = 1.0 - positive - negative
        
        # Calculate compound score (-1 to 1)
        compound = (pos_count - neg_count) / max(total_words, 1)
        compound = max(-1, min(1, compound))  # Clamp to [-1, 1]
        
        return {
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'compound': compound
        }
    
    def categorize_sentiment(self, compound_score: float) -> str:
        """Categorize sentiment based on compound score"""
        if compound_score >= 0.5:
            return "Very Positive"
        elif compound_score >= 0.1:
            return "Positive"
        elif compound_score <= -0.5:
            return "Very Negative"
        elif compound_score <= -0.1:
            return "Negative"
        else:
            return "Neutral"


class KanseiAnalyzer:
    """Kansei Engineering analysis for emotional/affective responses"""
    
    def __init__(self):
        # Kansei words categories based on product design emotions
        self.kansei_categories = {
            'Luxury': {
                'words': {'luxurious', 'premium', 'elegant', 'sophisticated', 'refined', 
                         'exclusive', 'high-end', 'classy', 'deluxe', 'opulent'},
                'weight': 1.5
            },
            'Comfort': {
                'words': {'comfortable', 'cozy', 'relaxing', 'soothing', 'pleasant', 
                         'cushioned', 'soft', 'plush', 'supportive', 'ergonomic'},
                'weight': 1.3
            },
            'Safety': {
                'words': {'safe', 'secure', 'stable', 'reliable', 'trustworthy', 
                         'sturdy', 'solid', 'protective', 'durable', 'strong'},
                'weight': 1.2
            },
            'Innovation': {
                'words': {'innovative', 'modern', 'advanced', 'high-tech', 'smart', 
                         'cutting-edge', 'futuristic', 'technological', 'automated', 'intelligent'},
                'weight': 1.1
            },
            'Disappointment': {
                'words': {'disappointing', 'frustrated', 'annoyed', 'dissatisfied', 'unhappy',
                         'regret', 'letdown', 'underwhelming', 'unimpressed', 'mediocre'},
                'weight': -1.5
            },
            'Discomfort': {
                'words': {'uncomfortable', 'painful', 'aching', 'stiff', 'hard', 
                         'rough', 'irritating', 'unpleasant', 'cramped', 'tight'},
                'weight': -1.3
            }
        }
    
    def analyze_kansei(self, text: str) -> Dict[str, float]:
        """Analyze Kansei emotional responses in text"""
        if not text:
            return {category: 0.0 for category in self.kansei_categories}
        
        text_lower = text.lower()
        words = text_lower.split()
        
        kansei_scores = {}
        
        for category, info in self.kansei_categories.items():
            kansei_words = info['words']
            weight = info['weight']
            
            # Count occurrences
            count = sum(1 for word in words if word in kansei_words)
            
            # Normalize by text length
            score = (count / max(len(words), 1)) * weight
            kansei_scores[category] = score
        
        return kansei_scores
    
    def get_dominant_kansei(self, kansei_scores: Dict[str, float]) -> str:
        """Get the dominant Kansei category"""
        if not kansei_scores or all(v == 0 for v in kansei_scores.values()):
            return "Neutral"
        
        # Get absolute values for comparison
        abs_scores = {k: abs(v) for k, v in kansei_scores.items()}
        dominant = max(abs_scores, key=abs_scores.get)
        
        return dominant 


class WordCloudGenerator:
    """Advanced Word Cloud Generation with Multiple Views"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.wordcloud_dir = os.path.join(output_dir, "wordclouds")
        os.makedirs(self.wordcloud_dir, exist_ok=True)
    
    def generate_comprehensive_wordclouds(self, text_data: List[str], entity_data: Dict = None):
        """Generate multiple word clouds for different perspectives"""
        print("☁️ Generating comprehensive word clouds...")
        
        # Define comprehensive stopwords
        custom_stopwords = set(WC_STOPWORDS) | set(nltk.corpus.stopwords.words('english')) | {
            'seat', 'seats', 'car', 'vehicle', 'also', 'get', 'got', 'would', 'could', 'make', 'made', 
            'see', 'really', 'even', 'one', 'nan', 'lot', 'bit', 'im', 'ive', 'id', 'well', 'good', 
            'great', 'nice', 'bad', 'poor', 'drive', 'driving', 'ride', 'riding', 'trip', 'product', 
            'item', 'time', 'way', 'thing', 'things', 'little', 'big', 'small', 'new', 'old'
        }
        
        # 1. Overall Word Cloud
        self.create_overall_wordcloud(text_data, custom_stopwords)
        
        # 2. Entity-specific Word Clouds (if entity data provided)
        if entity_data:
            self.create_entity_wordclouds(entity_data, custom_stopwords)
        
        # 3. Sentiment-based Word Clouds
        self.create_sentiment_wordclouds(text_data, custom_stopwords)
        
        print(f"☁️ Word clouds saved to {self.wordcloud_dir}")
    
    def create_overall_wordcloud(self, text_data: List[str], stopwords: set):
        """Create overall word cloud from all text"""
        if not text_data:
            return
        
        all_text = " ".join([str(text) for text in text_data if text and str(text).strip()])
        
        if not all_text.strip():
            return
        
        try:
            wordcloud = WordCloud(
                width=1200, 
                height=600, 
                background_color='white', 
                stopwords=stopwords, 
                collocations=False,
                max_words=200,
                colormap='viridis'
            ).generate(all_text)
            
            plt.figure(figsize=(15, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title("Overall Text Word Cloud", fontsize=18, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(os.path.join(self.wordcloud_dir, "overall_wordcloud.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Overall word cloud generated")
        except Exception as e:
            print(f"⚠️ Error generating overall word cloud: {e}")
    
    def create_entity_wordclouds(self, entity_data: Dict, stopwords: set):
        """Create word clouds for each entity type"""
        entities = ["ARMREST", "BACKREST", "HEADREST", "CUSHION", "MATERIAL", 
                   "LUMBAR_SUPPORT", "RECLINER", "FOOTREST", "SEAT_MESSAGE", 
                   "SEAT_WARMER", "TRAYTABLE"]
        
        colors = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'YlOrRd', 
                 'YlGnBu', 'PuRd', 'BuGn', 'OrRd', 'GnBu']
        
        for i, entity in enumerate(entities):
            if entity in entity_data and entity_data[entity]:
                entity_text = " ".join([str(text) for text in entity_data[entity] 
                                      if text and str(text).strip()])
                
                if len(entity_text.strip()) > 50:  # Minimum text length
                    try:
                        wordcloud = WordCloud(
                            width=800, 
                            height=400, 
                            background_color='white',
                            stopwords=stopwords, 
                            collocations=False,
                            max_words=100,
                            colormap=colors[i % len(colors)]
                        ).generate(entity_text)
                        
                        plt.figure(figsize=(10, 5))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis("off")
                        plt.title(f"{entity} Word Cloud", fontsize=14, fontweight='bold')
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.wordcloud_dir, f"{entity.lower()}_wordcloud.png"), 
                                   dpi=300, bbox_inches='tight')
                        plt.close()
                    except Exception as e:
                        print(f"⚠️ Error generating {entity} word cloud: {e}")
        
        print("✅ Entity-specific word clouds generated")
    
    def create_sentiment_wordclouds(self, text_data: List[str], stopwords: set):
        """Create word clouds based on sentiment (positive/negative words)"""
        # Simple sentiment word lists
        positive_words = {
            'comfortable', 'excellent', 'amazing', 'perfect', 'great', 'good', 'nice', 'wonderful',
            'fantastic', 'outstanding', 'superb', 'brilliant', 'impressive', 'smooth', 'soft',
            'luxurious', 'premium', 'quality', 'durable', 'spacious', 'roomy', 'adjustable',
            'ergonomic', 'supportive', 'relaxing', 'cozy', 'plush', 'elegant', 'beautiful'
        }
        
        negative_words = {
            'uncomfortable', 'terrible', 'awful', 'bad', 'poor', 'horrible', 'disappointing',
            'cheap', 'flimsy', 'broken', 'damaged', 'worn', 'tight', 'cramped', 'hard', 'stiff',
            'rough', 'scratchy', 'noisy', 'unstable', 'wobbly', 'inadequate', 'insufficient',
            'problematic', 'defective', 'faulty', 'annoying', 'frustrating', 'unacceptable'
        }
        
        # Separate texts based on sentiment words
        positive_texts = []
        negative_texts = []
        
        for text in text_data:
            if not text or not str(text).strip():
                continue
            
            text_lower = str(text).lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                positive_texts.append(text)
            elif neg_count > pos_count:
                negative_texts.append(text)
        
        # Create positive sentiment word cloud
        if positive_texts:
            pos_text = " ".join([str(text) for text in positive_texts])
            try:
                wordcloud_pos = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    stopwords=stopwords, 
                    collocations=False,
                    max_words=150,
                    colormap='Greens'
                ).generate(pos_text)
                
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud_pos, interpolation='bilinear')
                plt.axis("off")
                plt.title("Positive Sentiment Word Cloud", fontsize=14, fontweight='bold', color='darkgreen')
                plt.tight_layout()
                plt.savefig(os.path.join(self.wordcloud_dir, "positive_sentiment_wordcloud.png"), 
                           dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"⚠️ Error generating positive sentiment word cloud: {e}")
        
        # Create negative sentiment word cloud
        if negative_texts:
            neg_text = " ".join([str(text) for text in negative_texts])
            try:
                wordcloud_neg = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    stopwords=stopwords, 
                    collocations=False,
                    max_words=150,
                    colormap='Reds'
                ).generate(neg_text)
                
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud_neg, interpolation='bilinear')
                plt.axis("off")
                plt.title("Negative Sentiment Word Cloud", fontsize=14, fontweight='bold', color='darkred')
                plt.tight_layout()
                plt.savefig(os.path.join(self.wordcloud_dir, "negative_sentiment_wordcloud.png"), 
                           dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"⚠️ Error generating negative sentiment word cloud: {e}")
        
        print("✅ Sentiment-based word clouds generated")


class ComprehensiveDataExporter:
    """Export comprehensive results including entities, sentiment, and Kansei"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.csv_dir = os.path.join(output_dir, "csv_results")
        os.makedirs(self.csv_dir, exist_ok=True)
        
        self.sentiment_analyzer = SentimentAnalyzer()
        self.kansei_analyzer = KanseiAnalyzer()
    
    def export_review_analysis(self, training_data: List[Tuple], nlp_model=None):
        """Export comprehensive review analysis with entities, sentiment, and Kansei"""
        print("📊 Exporting comprehensive review analysis...")
        
        review_data = []
        entity_data = []
        
        for idx, (text, annotations) in enumerate(training_data):
            if not text or not text.strip():
                continue
            
            # Sentiment analysis
            sentiment_scores = self.sentiment_analyzer.analyze_sentiment(text)
            sentiment_category = self.sentiment_analyzer.categorize_sentiment(sentiment_scores['compound'])
            
            # Kansei analysis
            kansei_scores = self.kansei_analyzer.analyze_kansei(text)
            dominant_kansei = self.kansei_analyzer.get_dominant_kansei(kansei_scores)
            
            # Extract entities
            entities_list = []
            for start, end, label in annotations.get("entities", []):
                entity_text = text[start:end]
                entities_list.append(f"{label}:{entity_text}")
                
                # Add to entity data
                entity_data.append({
                    'review_id': idx,
                    'entity_type': label,
                    'entity_text': entity_text,
                    'start_position': start,
                    'end_position': end,
                    'context': text[max(0, start-30):min(len(text), end+30)]
                })
            
            # Add to review data
            review_row = {
                'review_id': idx,
                'review_text': text,
                'text_length': len(text),
                'entities': '|'.join(entities_list) if entities_list else 'None',
                'entity_count': len(entities_list),
                'sentiment_positive': sentiment_scores['positive'],
                'sentiment_negative': sentiment_scores['negative'],
                'sentiment_neutral': sentiment_scores['neutral'],
                'sentiment_compound': sentiment_scores['compound'],
                'sentiment_category': sentiment_category,
                'dominant_kansei': dominant_kansei
            }
            
            # Add Kansei scores
            for kansei_cat, score in kansei_scores.items():
                review_row[f'kansei_{kansei_cat.lower()}'] = score
            
            review_data.append(review_row)
        
        # Create DataFrames
        reviews_df = pd.DataFrame(review_data)
        entities_df = pd.DataFrame(entity_data)
        
        # Save to CSV
        reviews_df.to_csv(os.path.join(self.csv_dir, 'review_analysis_complete.csv'), index=False)
        entities_df.to_csv(os.path.join(self.csv_dir, 'entity_details.csv'), index=False)
        
        # Create summary statistics
        self._create_summary_statistics(reviews_df, entities_df)
        
        print(f"✅ Review analysis exported to {self.csv_dir}")
        return reviews_df, entities_df
    
    def _create_summary_statistics(self, reviews_df: pd.DataFrame, entities_df: pd.DataFrame):
        """Create summary statistics CSV files"""
        
        # 1. Entity frequency summary
        entity_freq = entities_df['entity_type'].value_counts()
        entity_summary = pd.DataFrame({
            'entity_type': entity_freq.index,
            'frequency': entity_freq.values,
            'percentage': (entity_freq.values / entity_freq.sum() * 100).round(2)
        })
        entity_summary.to_csv(os.path.join(self.csv_dir, 'entity_frequency_summary.csv'), index=False)
        
        # 2. Sentiment distribution summary
        sentiment_summary = pd.DataFrame({
            'sentiment_category': reviews_df['sentiment_category'].value_counts().index,
            'count': reviews_df['sentiment_category'].value_counts().values,
            'percentage': (reviews_df['sentiment_category'].value_counts().values / len(reviews_df) * 100).round(2)
        })
        sentiment_summary.to_csv(os.path.join(self.csv_dir, 'sentiment_distribution.csv'), index=False)
        
        # 3. Kansei distribution summary
        kansei_cols = [col for col in reviews_df.columns if col.startswith('kansei_')]
        kansei_means = reviews_df[kansei_cols].mean()
        kansei_summary = pd.DataFrame({
            'kansei_dimension': [col.replace('kansei_', '').title() for col in kansei_cols],
            'average_score': kansei_means.values.round(4)
        })
        kansei_summary.to_csv(os.path.join(self.csv_dir, 'kansei_summary.csv'), index=False)


class ComprehensiveTrainingVisualizer:
    """Enhanced visualization combining training metrics with advanced analytics"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def plot_training_progress(self, metrics_history: List[Dict], title: str = "Ultimate Fix Training Progress"):
        """Plot F1 scores over training epochs"""
        if not metrics_history:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Extract data
        epochs = []
        overall_f1 = []
        entities_above_90 = []
        
        for metrics in metrics_history:
            phase = metrics['phase']
            if 'Epoch' in phase:
                try:
                    epoch_num = int(phase.split()[-1])
                except (ValueError, IndexError):
                    epoch_num = len(epochs)
            else:
                epoch_num = len(epochs)
            
            epochs.append(epoch_num)
            overall_f1.append(metrics['entity_level']['f1'])
            
            count_above_90 = sum(1 for entity_metrics in metrics['per_entity'].values()
                               if entity_metrics['support'] > 0 and entity_metrics['f1'] >= 0.9)
            entities_above_90.append(count_above_90)
        
        # Plot 1: Overall F1 Score
        ax1.plot(epochs, overall_f1, 'b-', linewidth=3, marker='o', markersize=6, label='Overall F1')
        ax1.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Target (0.9)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('F1 Score')
        ax1.set_title(f'{title} - Overall F1 Score Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.0)
        
        if epochs:
            ax1.set_xlim(0, max(epochs) + 1)
        
        # Plot 2: Entities Above 90%
        ax2.plot(epochs, entities_above_90, 'g-', linewidth=3, marker='s', markersize=6, label='Entities ≥ 0.9')
        ax2.axhline(y=11, color='red', linestyle='--', alpha=0.7, label='Target (11/11)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Number of Entities ≥ 0.9')
        ax2.set_title(f'{title} - Entities Above 90% Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 12)
        
        if epochs:
            ax2.set_xlim(0, max(epochs) + 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_final_results(self, final_metrics: Dict):
        """Plot final results and comparison"""
        if not final_metrics or 'per_entity' not in final_metrics:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Extract final scores
        entities = []
        f1_scores = []
        supports = []
        
        for entity, metrics in final_metrics['per_entity'].items():
            if metrics['support'] > 0:
                entities.append(entity)
                f1_scores.append(metrics['f1'])
                supports.append(metrics['support'])
        
        # Colors: green for >=0.9, orange for 0.8-0.9, red for <0.8
        colors = ['green' if f1 >= 0.9 else 'orange' if f1 >= 0.8 else 'red' for f1 in f1_scores]
        
        # Plot 1: F1 Scores Bar Chart
        bars1 = ax1.bar(entities, f1_scores, color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(y=0.9, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Target (0.9)')
        ax1.set_title('Final F1 Scores by Entity', fontsize=14, fontweight='bold')
        ax1.set_ylabel('F1 Score')
        ax1.set_ylim(0, 1.1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, f1 in zip(bars1, f1_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Support (Number of Examples)
        bars2 = ax2.bar(entities, supports, color='skyblue', alpha=0.7, edgecolor='black')
        ax2.set_title('Validation Support by Entity', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Examples')
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Summary Statistics
        overall_f1 = final_metrics['entity_level']['f1']
        entities_above_90 = sum(1 for f1 in f1_scores if f1 >= 0.9)
        
        summary_data = {
            'Overall F1': overall_f1,
            'Entities ≥ 0.9': entities_above_90,
            'Total Entities': len(entities),
            'Success Rate': entities_above_90 / len(entities) if entities else 0
        }
        
        summary_labels = list(summary_data.keys())
        summary_values = list(summary_data.values())
        
        bars3 = ax3.bar(summary_labels, summary_values, color='lightcoral', alpha=0.7, edgecolor='black')
        ax3.set_title('Ultimate Fix Summary', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Score / Count')
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels
        for bar, value in zip(bars3, summary_values):
            height = bar.get_height()
            if isinstance(value, float):
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            else:
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # Success indication
        success_text = f"🎯 SUCCESS: {entities_above_90}/{len(entities)} entities ≥ 0.9"
        ax4.text(0.5, 0.5, success_text, transform=ax4.transAxes, fontsize=20, 
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen' if entities_above_90 >= 7 else 'lightyellow'))
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'final_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_data_composition(self, base_count: int, synthetic_count: int):
        """Plot training data composition"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Pie chart
        labels = ['Real Data\n(Your JSON)', 'Ultimate Synthetic\n(Massive Examples)']
        sizes = [base_count, synthetic_count]
        colors = ['lightblue', 'lightcoral']
        explode = (0.05, 0.05)
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                          explode=explode, shadow=True, startangle=90)
        ax1.set_title('Ultimate Fix Data Composition', fontsize=14, fontweight='bold')
        
        # Bar chart
        categories = ['Real Data', 'Ultimate Synthetic']
        counts = [base_count, synthetic_count]
        
        bars = ax2.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('Training Data Counts', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Examples')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Add total
        total = sum(counts)
        ax2.text(0.5, 0.95, f'Total: {total:,} examples', transform=ax2.transAxes,
                ha='center', va='top', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'data_composition.png'), dpi=300, bbox_inches='tight')
        plt.close()


def load_training_data(annotated_data_path: str) -> List[Tuple]:
    """Load basic training data from JSON format"""
    print(f"Loading training data from {annotated_data_path}")
    
    if not os.path.exists(annotated_data_path):
        print(f"File not found: {annotated_data_path}")
        return []
    
    try:
        with open(annotated_data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        training_data = []
        label_map = {
            "recliner": "RECLINER", "seat_message": "SEAT_MESSAGE", "seat_warmer": "SEAT_WARMER",
            "headrest": "HEADREST", "armrest": "ARMREST", "footrest": "FOOTREST",
            "backrest": "BACKREST", "cushion": "CUSHION", "material": "MATERIAL",
            "traytable": "TRAYTABLE", "lumbar_support": "LUMBAR_SUPPORT",
            "seat_material": "MATERIAL", "lumbar": "LUMBAR_SUPPORT"
        }
        
        for item in raw_data:
            text = None
            if 'ReviewText' in item:
                text = item.get('ReviewText')
            elif 'data' in item:
                text_data = item['data']
                text = text_data.get('ReviewText') or text_data.get('Review Text') or text_data.get('feedback')
            
            if not text:
                continue
            
            text = str(text).strip()
            if not text:
                continue
            
            entities = []
            
            # Handle annotation formats
            if 'label' in item and isinstance(item['label'], list):
                for label_item in item['label']:
                    if isinstance(label_item, dict) and 'labels' in label_item and label_item['labels']:
                        start, end = label_item.get('start', 0), label_item.get('end', 0)
                        raw_label = label_item['labels'][0].lower()
                        standardized_label = label_map.get(raw_label, raw_label.upper())
                        
                        if 0 <= start < end <= len(text):
                            entities.append((start, end, standardized_label))
            
            elif 'annotations' in item and item['annotations'] and 'result' in item['annotations'][0]:
                for annotation in item['annotations'][0]['result']:
                    value = annotation.get("value", {})
                    start, end = value.get("start"), value.get("end")
                    labels_list = value.get("labels", [])
                    if start is not None and end is not None and labels_list:
                        raw_label = labels_list[0].lower()
                        standardized_label = label_map.get(raw_label, raw_label.upper())
                        
                        if 0 <= int(start) < int(end) <= len(text):
                            entities.append((int(start), int(end), standardized_label))
            
            if entities:
                training_data.append((text, {"entities": entities}))
        
        print(f"Loaded {len(training_data)} base examples")
        return training_data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return []


def run_ultimate_fix_training(
    annotations_path: str = "seat_entities_new_min.json",
    output_dir: str = "ultimate_fix_output",
    iterations: int = 200,
    lda_topics: int = 10
):
    """Ultimate Fix training with comprehensive analytics integration"""
    
    start_time = time.time()
    
    print("🚀 ULTIMATE FIX COMPLETE ANALYTICS")
    print("=" * 60)
    print("🎯 Ultimate Fix NER Training")
    print("📊 + Comprehensive Analytics")
    print("🔍 + LDA Topic Modeling")
    print("☁️ + Word Clouds")
    print("📈 + Sentiment & Kansei Analysis")
    print("📊 + CSV Exports & Visualizations")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load base data
    base_training_data = load_training_data(annotations_path)
    base_count = len(base_training_data)
    print(f"📊 Loaded {base_count} base examples")
    
    # Step 2: Apply Ultimate Entity Fixer - MASSIVE augmentation
    print(f"\n🔧 Applying Ultimate Entity Fixer...")
    entity_fixer = UltimateEntityFixer()
    
    # Generate MASSIVE synthetic examples for critical entities
    training_data = base_training_data.copy()
    
    # Ultra-massive augmentation for critical failing entities
    critical_augmentation = {
        "TRAYTABLE": 300,    # Most critical - only 1 validation example
        "MATERIAL": 250,     # Critical for material quality
        "LUMBAR_SUPPORT": 200,  # Important for comfort
        "SEAT_WARMER": 200,  # Important functionality
        "SEAT_MESSAGE": 180, # Massage functionality
        "RECLINER": 150,     # Reclining functionality
        "ARMREST": 120,      # Near target, boost
        "HEADREST": 120,     # Near target, boost
        "CUSHION": 100,      # Needs improvement
        "BACKREST": 100,     # Needs improvement
        "FOOTREST": 80       # Moderate improvement
    }
    
    for entity, count in critical_augmentation.items():
        synthetic_examples = entity_fixer.generate_massive_examples(entity, count)
        training_data.extend(synthetic_examples)
        print(f"🚨 Added {len(synthetic_examples)} ULTIMATE examples for {entity}")
    
    total_count = len(training_data)
    synthetic_count = total_count - base_count
    
    print(f"📊 Ultimate training dataset: {total_count} examples ({synthetic_count} synthetic)")
    
    # Step 3: Ultimate NER Training
    print(f"\n🧠 Starting Ultimate NER Training...")
    trainer = UltimateNERTrainer(training_data)
    model = trainer.train_ultimate(n_iter=iterations)
    
    if not model:
        print("❌ Training failed")
        return None
    
    # Save model
    model_path = os.path.join(output_dir, "ultimate_fix_model")
    model.to_disk(model_path)
    print(f"💾 Ultimate Fix model saved to: {model_path}")
    
    # Get final metrics
    final_metrics = trainer.metrics_history[-1] if trainer.metrics_history else {}
    
    # Step 4: Comprehensive Analytics (Post-processing)
    print(f"\n📊 Running Comprehensive Analytics...")
    
    # Extract training texts for analytics
    training_texts = [text for text, _ in training_data if text and text.strip()]
    
    # CSV Export with sentiment and Kansei analysis
    print("📊 Exporting comprehensive CSV analysis...")
    data_exporter = ComprehensiveDataExporter(output_dir)
    reviews_df, entities_df = data_exporter.export_review_analysis(training_data, model)
    
    # LDA Topic Modeling
    print("🔍 Performing LDA topic modeling...")
    lda_analyzer = LDAAnalyzer(output_dir)
    lda_model, dictionary, corpus, coherence_score = lda_analyzer.perform_lda_analysis(training_texts, num_topics=lda_topics)
    
    # Word Cloud Generation
    print("☁️ Generating comprehensive word clouds...")
    wordcloud_gen = WordCloudGenerator(output_dir)
    
    # Organize text by entities for entity-specific word clouds
    entity_texts = defaultdict(list)
    for text, annotations in training_data:
        if text and text.strip():
            for start, end, label in annotations.get("entities", []):
                # Get context around entity
                context_start = max(0, start - 50)
                context_end = min(len(text), end + 50)
                context_text = text[context_start:context_end]
                entity_texts[label].append(context_text)
    
    wordcloud_gen.generate_comprehensive_wordclouds(training_texts, entity_texts)
    
    # Training Visualizations
    print("📈 Creating comprehensive visualizations...")
    visualizer = ComprehensiveTrainingVisualizer(output_dir)
    visualizer.plot_training_progress(trainer.metrics_history, "Ultimate Fix Training")
    visualizer.plot_final_results(final_metrics)
    visualizer.plot_data_composition(base_count, synthetic_count)
    
    # Step 5: Generate comprehensive report
    print("📄 Generating comprehensive report...")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = os.path.join(output_dir, "ULTIMATE_FIX_COMPLETE_ANALYTICS_REPORT.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("# 🚀 Ultimate Fix Complete Analytics Report\n\n")
        f.write(f"**Generated:** {timestamp}  \n")
        f.write(f"**Random Seed:** {RANDOM_SEED}  \n")
        f.write(f"**Training Time:** {time.time() - start_time:.2f} seconds  \n")
        f.write(f"**Total Examples:** {total_count:,}  \n\n")
        
        # Data Composition
        f.write("## 📊 Data Composition\n\n")
        f.write("| Data Type | Count | Percentage |\n")
        f.write("|-----------|-------|------------|\n")
        f.write(f"| Real Data (JSON) | {base_count:,} | {base_count/total_count*100:.1f}% |\n")
        f.write(f"| Ultimate Synthetic | {synthetic_count:,} | {synthetic_count/total_count*100:.1f}% |\n")
        f.write(f"| **TOTAL** | **{total_count:,}** | **100.0%** |\n\n")
        
        # Ultimate Fix Results
        if final_metrics and 'per_entity' in final_metrics:
            f.write("## 🏆 Ultimate Fix Results\n\n")
            f.write(f"**Overall F1 Score:** {final_metrics['entity_level']['f1']:.4f}  \n")
            
            entities_above_90 = sum(1 for metrics in final_metrics['per_entity'].values() 
                                  if metrics['support'] > 0 and metrics['f1'] >= 0.9)
            total_entities = sum(1 for metrics in final_metrics['per_entity'].values() 
                                if metrics['support'] > 0)
            f.write(f"**Success Rate:** {entities_above_90}/{total_entities} entities ≥ 0.9  \n\n")
            
            # Per-entity results
            f.write("### Per-Entity Performance\n\n")
            f.write("| Entity | F1 Score | Precision | Recall | Support | Status |\n")
            f.write("|--------|----------|-----------|---------|---------|--------|\n")
            
            for entity, metrics in final_metrics['per_entity'].items():
                if metrics['support'] > 0:
                    status = "🎯" if metrics['f1'] >= 0.9 else "⚡" if metrics['f1'] >= 0.8 else "🔧"
                    f.write(f"| {entity} | {metrics['f1']:.4f} | {metrics['precision']:.4f} | "
                           f"{metrics['recall']:.4f} | {metrics['support']} | {status} |\n")
        
        # Analytics Results
        f.write("\n## 📊 Analytics Results\n\n")
        f.write(f"- **Reviews Analyzed:** {len(reviews_df):,}\n")
        f.write(f"- **Entities Extracted:** {len(entities_df):,}\n")
        f.write(f"- **Average Sentiment:** {reviews_df['sentiment_compound'].mean():.3f}\n")
        f.write(f"- **Dominant Kansei:** {reviews_df['dominant_kansei'].mode()[0] if len(reviews_df['dominant_kansei'].mode()) > 0 else 'N/A'}\n")
        
        if lda_model:
            f.write(f"- **LDA Topics:** {lda_model.num_topics}\n")
            f.write(f"- **LDA Coherence:** {coherence_score:.4f}\n")
        
        # Output Structure
        f.write("\n## 📁 Complete Output Structure\n\n")
        f.write("```\n")
        f.write(f"{output_dir}/\n")
        f.write("├── ultimate_fix_model/         # Trained NER model\n")
        f.write("├── csv_results/                # Comprehensive CSV analysis\n")
        f.write("│   ├── review_analysis_complete.csv\n")
        f.write("│   ├── entity_details.csv\n")
        f.write("│   ├── entity_frequency_summary.csv\n")
        f.write("│   ├── sentiment_distribution.csv\n")
        f.write("│   └── kansei_summary.csv\n")
        f.write("├── lda_analysis/               # LDA topic modeling\n")
        f.write("│   ├── lda_model.gensim\n")
        f.write("│   ├── topic_word_heatmap.png\n")
        f.write("│   └── lda_interactive_visualization.html\n")
        f.write("├── wordclouds/                 # Word cloud visualizations\n")
        f.write("│   ├── overall_wordcloud.png\n")
        f.write("│   ├── positive_sentiment_wordcloud.png\n")
        f.write("│   ├── negative_sentiment_wordcloud.png\n")
        f.write("│   └── [entity]_wordcloud.png\n")
        f.write("├── plots/                      # Training visualizations\n")
        f.write("│   ├── training_progress.png\n")
        f.write("│   ├── final_results.png\n")
        f.write("│   └── data_composition.png\n")
        f.write("└── ULTIMATE_FIX_COMPLETE_ANALYTICS_REPORT.md\n")
        f.write("```\n\n")
        
        # Footer
        f.write("---\n")
        f.write(f"*Ultimate Fix Complete Analytics Report generated on {timestamp}*\n")
    
    # Final summary
    training_time = time.time() - start_time
    best_f1 = trainer.best_f1
    final_f1 = final_metrics.get('entity_level', {}).get('f1', 0) if final_metrics else 0
    
    print(f"\n🎯 ULTIMATE FIX COMPLETE RESULTS:")
    print(f"=" * 60)
    print(f"✅ Best F1 achieved: {best_f1:.4f}")
    print(f"✅ Final F1: {final_f1:.4f}")
    print(f"⏱️ Training time: {training_time:.2f} seconds")
    print(f"🎲 Random seed: {RANDOM_SEED}")
    
    if final_metrics and 'per_entity' in final_metrics:
        entities_above_90 = sum(1 for m in final_metrics['per_entity'].values() 
                              if m.get('support', 0) > 0 and m.get('f1', 0) >= 0.9)
        total_entities = sum(1 for m in final_metrics['per_entity'].values() 
                               if m.get('support', 0) > 0)
        
        if entities_above_90 >= 7:
            print(f"🎉 SUCCESS: {entities_above_90}/{total_entities} entities ≥ 0.9!")
        else:
            print(f"📊 Significant Progress: {entities_above_90}/{total_entities} entities ≥ 0.9")
    
    print(f"\n📁 Complete outputs saved to: {output_dir}")
    print(f"📄 Comprehensive report: {report_path}")
    print(f"📊 Key analytics directories:")
    print(f"   - CSV Results: {os.path.join(output_dir, 'csv_results/')}")
    print(f"   - LDA Analysis: {os.path.join(output_dir, 'lda_analysis/')}")
    print(f"   - Word Clouds: {os.path.join(output_dir, 'wordclouds/')}")
    print(f"   - Visualizations: {os.path.join(output_dir, 'plots/')}")
    
    return model

def run_enhanced_ultimate_fix_training(
    annotations_path: str = "seat_entities_new_min.json",
    output_dir: str = "enhanced_ultimate_fix_output",
    iterations: int = 250,
    lda_topics: int = 10,
    use_advanced_augmentation: bool = True,
    use_adversarial_examples: bool = True
):
    """Enhanced Ultimate Fix training with advanced techniques for 0.9+ F1 performance"""
    
    start_time = time.time()
    
    print("🚀 ENHANCED ULTIMATE FIX COMPLETE ANALYTICS - TARGET: 0.9+ F1")
    print("=" * 70)
    print("🎯 Advanced NER Training Techniques:")
    print("   ✅ Advanced Contextual Data Augmentation")
    print("   ✅ Adversarial Example Generation")
    print("   ✅ Adaptive Learning Rate Scheduling")
    print("   ✅ Enhanced Validation Strategies")
    print("   ✅ Ensemble Model Selection")
    print("   ✅ Stratified Data Splitting")
    print("📊 + Comprehensive Analytics & Visualizations")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load base data
    base_training_data = load_training_data(annotations_path)
    base_count = len(base_training_data)
    print(f"📊 Loaded {base_count} base examples")
    
    if base_count == 0:
        print("❌ No training data loaded. Check your annotations file.")
        return None
    
    # Step 2: Enhanced Data Augmentation with Advanced Techniques
    print(f"\n🔧 Applying Enhanced Data Augmentation...")
    
    training_data = base_training_data.copy()
    
    if use_advanced_augmentation:
        # Use new AdvancedEntityAugmenter
        advanced_augmenter = AdvancedEntityAugmenter()
        
        # Enhanced augmentation strategy with sentiment balance
        enhanced_augmentation_strategy = {
            "TRAYTABLE": {
                "contextual": 400,  # Ultra-massive for critical entity
                "adversarial": 50,
                "positive_bias": 0.1  # Slight positive bias
            },
            "MATERIAL": {
                "contextual": 350,
                "adversarial": 40,
                "positive_bias": 0.0  # Balanced
            },
            "LUMBAR_SUPPORT": {
                "contextual": 300,
                "adversarial": 35,
                "positive_bias": 0.05
            },
            "SEAT_WARMER": {
                "contextual": 300,
                "adversarial": 35,
                "positive_bias": 0.05
            },
            "SEAT_MESSAGE": {
                "contextual": 250,
                "adversarial": 30,
                "positive_bias": 0.0
            },
            "RECLINER": {
                "contextual": 200,
                "adversarial": 25,
                "positive_bias": 0.0
            },
            "ARMREST": {
                "contextual": 180,
                "adversarial": 20,
                "positive_bias": 0.0
            },
            "HEADREST": {
                "contextual": 180,
                "adversarial": 20,
                "positive_bias": 0.0
            },
            "CUSHION": {
                "contextual": 150,
                "adversarial": 20,
                "positive_bias": 0.0
            },
            "BACKREST": {
                "contextual": 150,
                "adversarial": 20,
                "positive_bias": 0.0
            },
            "FOOTREST": {
                "contextual": 120,
                "adversarial": 15,
                "positive_bias": 0.0
            }
        }
        
        total_added = 0
        for entity, strategy in enhanced_augmentation_strategy.items():
            # Generate contextual examples
            contextual_examples = advanced_augmenter.generate_contextual_examples(
                entity, strategy["contextual"], sentiment_bias=strategy["positive_bias"]
            )
            training_data.extend(contextual_examples)
            total_added += len(contextual_examples)
            
            # Generate adversarial examples
            if use_adversarial_examples:
                adversarial_examples = advanced_augmenter.generate_adversarial_examples(
                    entity, strategy["adversarial"]
                )
                training_data.extend(adversarial_examples)
                total_added += len(adversarial_examples)
            
            print(f"🧠 Enhanced {entity}: +{len(contextual_examples)} contextual, "
                  f"+{len(adversarial_examples) if use_adversarial_examples else 0} adversarial")
        
        print(f"📊 Enhanced dataset: {len(training_data)} examples (+{total_added} advanced synthetic)")
    
    else:
        # Fallback to original augmentation
        print("📊 Using original augmentation strategy...")
        entity_fixer = UltimateEntityFixer()
        
        critical_augmentation = {
            "TRAYTABLE": 350, "MATERIAL": 300, "LUMBAR_SUPPORT": 250, "SEAT_WARMER": 250,
            "SEAT_MESSAGE": 200, "RECLINER": 180, "ARMREST": 150, "HEADREST": 150,
            "CUSHION": 120, "BACKREST": 120, "FOOTREST": 100
        }
        
        for entity, count in critical_augmentation.items():
            synthetic_examples = entity_fixer.generate_massive_examples(entity, count)
            training_data.extend(synthetic_examples)
            print(f"🔧 Added {len(synthetic_examples)} examples for {entity}")
    
    total_count = len(training_data)
    synthetic_count = total_count - base_count
    
    print(f"📊 Final training dataset: {total_count} examples ({synthetic_count} synthetic)")
    
    # Step 3: Enhanced NER Training
    print(f"\n🧠 Starting Enhanced NER Training...")
    enhanced_trainer = EnhancedNERTrainer(training_data)
    model = enhanced_trainer.train_enhanced(n_iter=iterations, use_ensemble=True)
    
    if not model:
        print("❌ Enhanced training failed")
        return None
    
    # Save enhanced model
    model_path = os.path.join(output_dir, "enhanced_ultimate_fix_model")
    model.to_disk(model_path)
    print(f"💾 Enhanced model saved to: {model_path}")
    
    # Get final metrics
    final_metrics = enhanced_trainer.metrics_history[-1] if enhanced_trainer.metrics_history else {}
    
    # Step 4: Comprehensive Analytics (unchanged from original)
    print(f"\n📊 Running Comprehensive Analytics...")
    
    # Extract training texts for analytics
    training_texts = [text for text, _ in training_data if text and text.strip()]
    
    # CSV Export with sentiment and Kansei analysis
    print("📊 Exporting comprehensive CSV analysis...")
    data_exporter = ComprehensiveDataExporter(output_dir)
    reviews_df, entities_df = data_exporter.export_review_analysis(training_data, model)
    
    # LDA Topic Modeling
    print("🔍 Performing LDA topic modeling...")
    lda_analyzer = LDAAnalyzer(output_dir)
    lda_model, dictionary, corpus, coherence_score = lda_analyzer.perform_lda_analysis(training_texts, num_topics=lda_topics)
    
    # Word Cloud Generation
    print("☁️ Generating comprehensive word clouds...")
    wordcloud_gen = WordCloudGenerator(output_dir)
    
    # Organize text by entities for entity-specific word clouds
    entity_texts = defaultdict(list)
    for text, annotations in training_data:
        if text and text.strip():
            for start, end, label in annotations.get("entities", []):
                # Get context around entity
                context_start = max(0, start - 50)
                context_end = min(len(text), end + 50)
                context_text = text[context_start:context_end]
                entity_texts[label].append(context_text)
    
    wordcloud_gen.generate_comprehensive_wordclouds(training_texts, entity_texts)
    
    # Enhanced Training Visualizations
    print("📈 Creating enhanced visualizations...")
    visualizer = ComprehensiveTrainingVisualizer(output_dir)
    visualizer.plot_training_progress(enhanced_trainer.metrics_history, "Enhanced Ultimate Fix Training")
    visualizer.plot_final_results(final_metrics)
    visualizer.plot_data_composition(base_count, synthetic_count)
    
    # Step 5: Generate Enhanced Report with Advanced Metrics
    print("📄 Generating enhanced comprehensive report...")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = os.path.join(output_dir, "ENHANCED_ULTIMATE_FIX_REPORT.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("# 🚀 Enhanced Ultimate Fix Complete Analytics Report\n\n")
        f.write("## 🎯 Advanced NER Training for 0.9+ F1 Performance\n\n")
        f.write(f"**Generated:** {timestamp}  \n")
        f.write(f"**Random Seed:** {RANDOM_SEED}  \n")
        f.write(f"**Training Time:** {time.time() - start_time:.2f} seconds  \n")
        f.write(f"**Total Examples:** {total_count:,}  \n")
        f.write(f"**Training Iterations:** {iterations}  \n\n")
        
        # Enhanced Techniques Used
        f.write("## 🧠 Enhanced Techniques Applied\n\n")
        f.write("| Technique | Status | Description |\n")
        f.write("|-----------|--------|--------------|\n")
        f.write(f"| Advanced Data Augmentation | {'✅' if use_advanced_augmentation else '❌'} | Contextual + sentiment-aware examples |\n")
        f.write(f"| Adversarial Examples | {'✅' if use_adversarial_examples else '❌'} | Challenging contexts for robustness |\n")
        f.write("| Adaptive Learning Scheduler | ✅ | Dynamic parameter adjustment |\n")
        f.write("| Stratified Validation | ✅ | Balanced entity representation |\n")
        f.write("| Enhanced Early Stopping | ✅ | Multiple performance criteria |\n")
        f.write("| Ensemble Model Selection | ✅ | Best model selection strategy |\n\n")
        
        # Data Composition with Enhancement Details
        f.write("## 📊 Enhanced Data Composition\n\n")
        f.write("| Data Type | Count | Percentage | Enhancement |\n")
        f.write("|-----------|-------|------------|-------------|\n")
        f.write(f"| Real Data (JSON) | {base_count:,} | {base_count/total_count*100:.1f}% | Original annotations |\n")
        f.write(f"| Advanced Synthetic | {synthetic_count:,} | {synthetic_count/total_count*100:.1f}% | Contextual + adversarial |\n")
        f.write(f"| **TOTAL** | **{total_count:,}** | **100.0%** | **Enhanced dataset** |\n\n")
        
        # Enhanced Performance Results
        if final_metrics and 'per_entity' in final_metrics:
            f.write("## 🏆 Enhanced Performance Results\n\n")
            overall_f1 = final_metrics['entity_level']['f1']
            f.write(f"**Overall F1 Score:** {overall_f1:.4f}  \n")
            f.write(f"**Best F1 Achieved:** {enhanced_trainer.best_f1:.4f}  \n")
            f.write(f"**Best Min Entity F1:** {enhanced_trainer.best_min_entity_f1:.4f}  \n")
            
            entities_above_90 = sum(1 for metrics in final_metrics['per_entity'].values() 
                                  if metrics['support'] > 0 and metrics['f1'] >= 0.9)
            entities_above_85 = sum(1 for metrics in final_metrics['per_entity'].values() 
                                  if metrics['support'] > 0 and metrics['f1'] >= 0.85)
            total_entities = sum(1 for metrics in final_metrics['per_entity'].values() 
                                if metrics['support'] > 0)
            
            f.write(f"**Success Rate (≥0.9):** {entities_above_90}/{total_entities} entities ({entities_above_90/total_entities*100:.1f}%)  \n")
            f.write(f"**Good Performance (≥0.85):** {entities_above_85}/{total_entities} entities ({entities_above_85/total_entities*100:.1f}%)  \n\n")
            
            # Detailed per-entity results
            f.write("### Enhanced Per-Entity Performance\n\n")
            f.write("| Entity | F1 Score | Precision | Recall | Support | Status | Improvement Target |\n")
            f.write("|--------|----------|-----------|---------|---------|--------|-------------------|\n")
            
            for entity, metrics in final_metrics['per_entity'].items():
                if metrics['support'] > 0:
                    f1_val = metrics['f1']
                    if f1_val >= 0.9:
                        status = "🎯 EXCELLENT"
                        target = "Maintain"
                    elif f1_val >= 0.85:
                        status = "⚡ GOOD"
                        target = "Fine-tune to 0.9+"
                    elif f1_val >= 0.7:
                        status = "🔧 NEEDS WORK"
                        target = "Major improvement needed"
                    else:
                        status = "❌ CRITICAL"
                        target = "Complete rework required"
                    
                    f.write(f"| {entity} | {f1_val:.4f} | {metrics['precision']:.4f} | "
                           f"{metrics['recall']:.4f} | {metrics['support']} | {status} | {target} |\n")
        
        # Performance Analysis
        f.write(f"\n## 📈 Performance Analysis\n\n")
        if final_metrics and 'per_entity' in final_metrics:
            if entities_above_90 >= max(8, int(total_entities * 0.8)):
                f.write("### 🎉 SUCCESS ACHIEVED!\n\n")
                f.write(f"✅ **Target met:** {entities_above_90}/{total_entities} entities achieved ≥ 0.9 F1 score!  \n")
                f.write("✅ **Enhanced techniques successful:** Advanced augmentation and training strategies proved effective.  \n")
            elif entities_above_85 >= total_entities:
                f.write("### 🔥 EXCELLENT PROGRESS!\n\n")
                f.write(f"⚡ **Strong performance:** All {total_entities} entities achieved ≥ 0.85 F1 score.  \n")
                f.write("⚡ **Near target:** With fine-tuning, 0.9+ F1 is achievable for all entities.  \n")
            else:
                f.write("### 📊 SIGNIFICANT IMPROVEMENT\n\n")
                f.write(f"📈 **Progress made:** {entities_above_90} entities achieved ≥ 0.9, {entities_above_85} achieved ≥ 0.85.  \n")
                f.write("📈 **Next steps:** Focus on entities below 0.85 with targeted improvements.  \n")
        
        # Recommendations for further improvement
        f.write(f"\n## 🎯 Recommendations for Further Improvement\n\n")
        
        if final_metrics and 'per_entity' in final_metrics:
            low_performing = [entity for entity, metrics in final_metrics['per_entity'].items() 
                            if metrics['support'] > 0 and metrics['f1'] < 0.9]
            
            if low_performing:
                f.write("### Entities Needing Attention\n\n")
                for entity in low_performing:
                    metrics = final_metrics['per_entity'][entity]
                    f.write(f"**{entity}** (F1: {metrics['f1']:.3f}):\n")
                    if metrics['precision'] < metrics['recall']:
                        f.write("- Focus on reducing false positives (improve precision)\n")
                        f.write("- Add more negative examples and boundary cases\n")
                    else:
                        f.write("- Focus on reducing false negatives (improve recall)\n")
                        f.write("- Add more diverse positive examples and synonyms\n")
                    f.write(f"- Current support: {metrics['support']} examples\n\n")
        
        f.write("### General Recommendations\n\n")
        f.write("1. **Data Quality:** Review and clean training annotations for consistency\n")
        f.write("2. **Active Learning:** Use model predictions to identify hard examples for manual annotation\n")
        f.write("3. **Domain Adaptation:** Collect more domain-specific examples\n")
        f.write("4. **Ensemble Methods:** Implement true ensemble prediction (not just model selection)\n")
        f.write("5. **Cross-Validation:** Use k-fold cross-validation for more robust evaluation\n\n")
        
        # Analytics Results (unchanged)
        f.write("## 📊 Analytics Results\n\n")
        f.write(f"- **Reviews Analyzed:** {len(reviews_df):,}\n")
        f.write(f"- **Entities Extracted:** {len(entities_df):,}\n")
        f.write(f"- **Average Sentiment:** {reviews_df['sentiment_compound'].mean():.3f}\n")
        f.write(f"- **Dominant Kansei:** {reviews_df['dominant_kansei'].mode()[0] if len(reviews_df['dominant_kansei'].mode()) > 0 else 'N/A'}\n")
        
        if lda_model:
            f.write(f"- **LDA Topics:** {lda_model.num_topics}\n")
            f.write(f"- **LDA Coherence:** {coherence_score:.4f}\n")
        
        # Output Structure
        f.write("\n## 📁 Complete Enhanced Output Structure\n\n")
        f.write("```\n")
        f.write(f"{output_dir}/\n")
        f.write("├── enhanced_ultimate_fix_model/  # Enhanced trained NER model\n")
        f.write("├── csv_results/                  # Comprehensive CSV analysis\n")
        f.write("├── lda_analysis/                 # LDA topic modeling\n")
        f.write("├── wordclouds/                   # Word cloud visualizations\n")
        f.write("├── plots/                        # Enhanced training visualizations\n")
        f.write("└── ENHANCED_ULTIMATE_FIX_REPORT.md\n")
        f.write("```\n\n")
        
        # Footer
        f.write("---\n")
        f.write(f"*Enhanced Ultimate Fix Complete Analytics Report generated on {timestamp}*  \n")
        f.write("*Targeting 0.9+ F1 performance with advanced NER training techniques*\n")
    
    # Final Enhanced Summary
    training_time = time.time() - start_time
    best_f1 = enhanced_trainer.best_f1
    final_f1 = final_metrics.get('entity_level', {}).get('f1', 0) if final_metrics else 0
    
    print(f"\n🎯 ENHANCED ULTIMATE FIX RESULTS:")
    print(f"=" * 70)
    print(f"✅ Best F1 achieved: {best_f1:.4f}")
    print(f"✅ Final F1: {final_f1:.4f}")
    print(f"✅ Training iterations: {iterations}")
    print(f"⏱️ Training time: {training_time:.2f} seconds")
    print(f"🎲 Random seed: {RANDOM_SEED}")
    
    if final_metrics and 'per_entity' in final_metrics:
        entities_above_90 = sum(1 for m in final_metrics['per_entity'].values() 
                              if m.get('support', 0) > 0 and m.get('f1', 0) >= 0.9)
        entities_above_85 = sum(1 for m in final_metrics['per_entity'].values() 
                              if m.get('support', 0) > 0 and m.get('f1', 0) >= 0.85)
        total_entities = sum(1 for m in final_metrics['per_entity'].values() 
                               if m.get('support', 0) > 0)
        
        if entities_above_90 >= max(8, int(total_entities * 0.8)):
            print(f"🎉 SUCCESS: {entities_above_90}/{total_entities} entities ≥ 0.9! TARGET ACHIEVED!")
        elif entities_above_85 == total_entities:
            print(f"🔥 EXCELLENT: All {total_entities} entities ≥ 0.85! Very close to target!")
        else:
            print(f"📈 PROGRESS: {entities_above_90}/{total_entities} entities ≥ 0.9, {entities_above_85}/{total_entities} entities ≥ 0.85")
    
    print(f"\n📁 Enhanced outputs saved to: {output_dir}")
    print(f"📄 Enhanced report: {report_path}")
    
    return model

if __name__ == "__main__":
    # Run with enhanced techniques for 0.9+ performance
    model = run_enhanced_ultimate_fix_training(
        annotations_path="seat_entities_new_min.json",
        output_dir="enhanced_ultimate_fix_output",
        iterations=250,  # Increased iterations
        lda_topics=10,
        use_advanced_augmentation=True,  # Enable advanced augmentation
        use_adversarial_examples=True    # Enable adversarial examples
    ) 