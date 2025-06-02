import pandas as pd
import re
import unicodedata
from bs4 import BeautifulSoup
import os
import random
import pathlib
import spacy
import json
import sys
import platform
import logging
import subprocess # Keep if system calls are absolutely needed, otherwise remove
from sklearn.model_selection import KFold, train_test_split # Keep if used, else remove
from spacy.tokens import DocBin
from spacy.training import Example
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from transformers import pipeline, AutoTokenizer
import warnings
import nltk
import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel, CoherenceModel # Added LdaModel for Kansei part
from gensim.models.phrases import Phrases, Phraser # Keep if used, else remove
import numpy as np
import string
import torch
from functools import lru_cache
from typing import List, Dict, Tuple, Optional
import gc
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
from wordcloud import WordCloud, STOPWORDS as WC_STOPWORDS
import pyLDAvis # For Kansei LDA visualization
import pyLDAvis.gensim_models as gensimvis # For Kansei LDA visualization

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Global Constants and Configurations ---
STANDARDIZED_LABELS = {
    "ARMREST", "BACKREST", "HEADREST", "CUSHION", "MATERIAL",
    "LUMBAR_SUPPORT", "SEAT_SIZE", "RECLINER", "FOOTREST",
    "SEAT_MESSAGE", "SEAT_WARMER", "TRAYTABLE" # Added from Kansei script
}

SEAT_SYNONYMS = { # Merged and expanded
    "ARMREST": ["armrest", "arm rest", "arm-rest", "armrests", "arm support", "elbow rest"],
    "BACKREST": ["backrest", "seat back", "seatback", "back", "back support", "back-rest", "spine support", "ergonomic back"],
    "HEADREST": ["headrest", "neckrest", "head support", "neck support", "head-rest", "headrests", "head cushion", "neck cushion"],
    "CUSHION": ["cushion", "padding", "cushioning", "seat base", "base", "bottom", "cushions", "padded", "pad", "memory foam", "seat cushion"],
    "MATERIAL": ["material", "fabric", "leather", "upholstery", "vinyl", "cloth", "velvet", "textile", "materials", "synthetic leather", "genuine leather", "premium leather", "suede", "canvas", "linen", "deer skin", "breathable fabric", "high-quality material"],
    "LUMBAR_SUPPORT": ["lumbar support", "lumbar", "lumbar pad", "lumbar cushion", "lower back support", "ergonomic lumbar", "adjustable lumbar", "spine alignment"],
    "SEAT_SIZE": ["legroom", "leg room", "space", "seat width", "seat size", "narrow", "tight", "cramped", "spacious", "roomy"],
    "RECLINER": ["reclining", "recline", "recliner", "reclined", "reclines", "reclinable", "seat angle", "seat position", "lie flat", "flat bed", "180 degree", "fully reclined", "tilting backrest"],
    "FOOTREST": ["footrest", "foot-rest", "footrests", "leg support", "ottoman", "leg extension", "calf support", "adjustable footrest"],
    "SEAT_MESSAGE": ["massage", "massaging", "massager", "massage function", "vibration", "vibrating", "therapeutic massage", "lumbar massage", "seat massage"],
    "SEAT_WARMER": ["warmer", "warming", "heated", "heating", "seat warmer", "seat heating", "temperature control", "warm seat", "climate control", "thermal comfort"],
    "TRAYTABLE": ["tray table", "fold down table", "dining table", "work table", "work surface", "laptop table", "laptop tray"]
}

# Download required NLTK data (from Kansei script, good practice)
try:
    nltk.data.find('corpora/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


# --- Optimized Helper Functions (from nlp_model_optimized.py) ---
@lru_cache(maxsize=None)
def ensure_directory(path: str) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

@lru_cache(maxsize=10000)
def clean_text_cached(txt: str) -> str:
    if pd.isna(txt) or not txt:
        return ""
    txt = BeautifulSoup(str(txt), "lxml").get_text(" ")
    txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode()
    txt = re.sub(r"\s+", " ", txt).strip()
    # Additional cleaning from Kansei script's preprocess_text
    txt = txt.replace('&', ' and ')
    txt = txt.replace('+', ' plus ')
    txt = txt.replace('@', ' at ')
    txt = re.sub(r'https?://\S+|www\.\S+', '', txt) # Remove URLs
    txt = re.sub(r'@\w+|#\w+', '', txt) # Remove mentions and hashtags
    abbreviations = {
        'e.g.': 'for example', 'i.e.': 'that is', 'etc.': 'and so on',
        'vs.': 'versus', 'approx.': 'approximately', 'min.': 'minimum', 'max.': 'maximum'
    }
    for abbr, full in abbreviations.items():
        txt = txt.replace(abbr, full)
    return txt.lower() # Consistent lowercasing

class PatternMatcher: # From nlp_model_optimized.py
    def __init__(self, seat_synonyms: Dict[str, List[str]]):
        self.patterns = {}
        for label, terms in seat_synonyms.items():
            pattern = r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b"
            self.patterns[label] = re.compile(pattern, re.IGNORECASE)

    @lru_cache(maxsize=5000)
    def find_matches(self, text: str) -> List[Tuple[int, int, str]]:
        spans = []
        text_lower = text.lower()
        nlp_blank = spacy.blank("en")
        doc = nlp_blank.make_doc(text)
        for label, pattern in self.patterns.items():
            for match in pattern.finditer(text_lower):
                start, end = match.start(), match.end()
                try:
                    span = doc.char_span(start, end, label=label, alignment_mode="contract")
                    if span is not None:
                        spans.append((start, end, label))
                    else:
                        adjusted_start, adjusted_end = start, end
                        for token in doc:
                            if token.idx <= start < token.idx + len(token.text):
                                adjusted_start = token.idx
                            if token.idx < end <= token.idx + len(token.text):
                                adjusted_end = token.idx + len(token.text)
                        if adjusted_start != start or adjusted_end != end:
                            adjusted_span = doc.char_span(adjusted_start, adjusted_end, label=label, alignment_mode="contract")
                            if adjusted_span is not None:
                                spans.append((adjusted_start, adjusted_end, label))
                except Exception:
                    continue
        return spans

def process_data_efficiently(csv_path: str, text_col: str) -> Tuple[pd.DataFrame, List[str]]:
    logger.info(f"Loading data from {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file '{csv_path}' not found")

    all_chunks = []
    test_data_texts = []
    chunk_size = 10000
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    df_reader = None

    for encoding in encodings:
        try:
            df_reader = pd.read_csv(csv_path, chunksize=chunk_size, encoding=encoding)
            logger.info(f"Successfully opened CSV with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.warning(f"Failed to read with {encoding}: {e}")
            continue

    if df_reader is None:
        raise ValueError("Could not read CSV file with any supported encoding")

    for chunk_idx, chunk in enumerate(df_reader):
        logger.info(f"Processing chunk {chunk_idx + 1}, size: {len(chunk)}")
        original_text_col = text_col # Store original for reference

        if text_col not in chunk.columns:
            possible_cols = [col for col in chunk.columns if any(keyword in col.lower() for keyword in ['review', 'text', 'feedback', 'comment'])]
            if possible_cols:
                text_col = possible_cols[0]
                logger.info(f"Using column '{text_col}' as text column for this chunk.")
            elif len(chunk.columns) == 1:
                text_col = chunk.columns[0]
                logger.info(f"Using only available column '{text_col}' as text column for this chunk.")
            else:
                logger.error(f"Text column '{original_text_col}' not found in chunk {chunk_idx + 1} and no suitable alternative. Columns: {list(chunk.columns)}")
                continue # Skip chunk or raise error

        chunk = chunk.dropna(subset=[text_col])
        if chunk.empty:
            logger.warning(f"Chunk {chunk_idx + 1} is empty after removing NaNs from text column.")
            continue

        chunk["clean_text"] = chunk[text_col].apply(lambda x: clean_text_cached(str(x)))
        chunk = chunk[chunk["clean_text"].str.len() > 0]

        if chunk.empty:
            logger.warning(f"Chunk {chunk_idx + 1} is empty after cleaning.")
            continue

        all_chunks.append(chunk)

        if len(chunk["clean_text"]) >= 10:
            test_size = max(1, min(int(0.1 * len(chunk["clean_text"])), 100))
            try:
                test_sample = random.sample(chunk["clean_text"].tolist(), test_size)
                test_data_texts.extend(test_sample)
            except ValueError as e:
                logger.warning(f"Could not sample test data from chunk: {e}")
        text_col = original_text_col # Reset for next chunk

    if not all_chunks:
        raise ValueError("No valid data found after processing all chunks.")

    full_df = pd.concat(all_chunks, ignore_index=True)
    test_data_texts = list(set(test_data_texts))
    logger.info(f"Processed {len(full_df):,} texts, {len(test_data_texts)} test samples")

    if full_df.empty:
        raise ValueError("No valid texts found in the dataset")

    return full_df, test_data_texts


def map_text_offsets(original_text: str, cleaned_text: str, start: int, end: int) -> Tuple[Optional[int], Optional[int]]:
    if not original_text or not cleaned_text: return None, None
    try:
        original_snippet = original_text[start:end].strip()
        cleaned_start = cleaned_text.lower().find(original_snippet.lower())
        if cleaned_start != -1:
            return cleaned_start, cleaned_start + len(original_snippet)

        # Fallback: character-by-character mapping (simplified)
        # This is a complex problem; a more robust solution would involve diffing
        # For now, if direct find fails, we might have issues.
        # A simple proportional mapping can be very inaccurate.
        logger.debug(f"Could not directly map snippet '{original_snippet}' in cleaned text. Offset mapping might be inaccurate.")
        return None, None # Prefer to return None if mapping is uncertain
    except Exception as e:
        logger.warning(f"Error mapping offsets for snippet '{original_text[start:end]}': {e}")
        return None, None

def load_training_data(annotated_data_path: str) -> List[Tuple]:
    logger.info(f"Loading training data from {annotated_data_path}")
    if not os.path.exists(annotated_data_path):
        logger.warning(f"Training data file '{annotated_data_path}' not found")
        return []
    try:
        with open(annotated_data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        training_data = []
        # Expanded label map from Kansei script's annotation structure
        label_map = {
            "recliner": "RECLINER", "seat_message": "SEAT_MESSAGE", "seat_warmer": "SEAT_WARMER",
            "headrest": "HEADREST", "armrest": "ARMREST", "footrest": "FOOTREST",
            "backrest": "BACKREST", "cushion": "CUSHION", "material": "MATERIAL",
            "traytable": "TRAYTABLE", "lumbar_support": "LUMBAR_SUPPORT",
            # Aliases from nlp_model_optimized
            "seat_material": "MATERIAL", "legroom": "SEAT_SIZE", "seat_size": "SEAT_SIZE",
            "seat size": "SEAT_SIZE", "lumbar": "LUMBAR_SUPPORT"
        }

        for item in raw_data:
            text = None
            # Handle different possible text keys from both annotation formats
            if 'ReviewText' in item: # Kansei format
                 text = item.get('ReviewText')
            elif 'data' in item and ('ReviewText' in item['data'] or 'Review Text' in item['data'] or 'feedback' in item['data']): # Label Studio format
                text_data = item['data']
                text = text_data.get('ReviewText') or text_data.get('Review Text') or text_data.get('feedback')

            if not text: continue
            cleaned_text = clean_text_cached(text) # Use the same cleaning
            if not cleaned_text: continue

            entities = []
            # Handle Kansei annotation structure
            if 'label' in item and isinstance(item['label'], list):
                for label_item in item['label']:
                    if isinstance(label_item, dict) and 'labels' in label_item and label_item['labels']:
                        start, end = label_item.get('start', 0), label_item.get('end', 0)
                        raw_label = label_item['labels'][0].lower()
                        standardized_label = label_map.get(raw_label)
                        if not standardized_label and raw_label.upper() in STANDARDIZED_LABELS: # Fallback if direct map fails
                            standardized_label = raw_label.upper()

                        if standardized_label:
                            mapped_start, mapped_end = map_text_offsets(text, cleaned_text, start, end)
                            if mapped_start is not None and mapped_end is not None and mapped_start < mapped_end:
                                entities.append((mapped_start, mapped_end, standardized_label))

            # Handle Label Studio annotation structure
            elif 'annotations' in item and item['annotations'] and 'result' in item['annotations'][0]:
                for annotation in item['annotations'][0]['result']:
                    value = annotation.get("value", {})
                    start, end = value.get("start"), value.get("end")
                    labels_list = value.get("labels", [])
                    if start is not None and end is not None and labels_list:
                        raw_label = labels_list[0].lower()
                        standardized_label = label_map.get(raw_label)
                        if not standardized_label and raw_label.upper() in STANDARDIZED_LABELS:
                             standardized_label = raw_label.upper()

                        if standardized_label:
                            mapped_start, mapped_end = map_text_offsets(text, cleaned_text, int(start), int(end))
                            if mapped_start is not None and mapped_end is not None and mapped_start < mapped_end:
                                entities.append((mapped_start, mapped_end, standardized_label))

            if entities:
                training_data.append((cleaned_text, {"entities": entities}))

        logger.info(f"Loaded {len(training_data)} training examples after processing and mapping.")
        if training_data:
            logger.info(f"Example training data item: Text='{training_data[0][0][:100]}...', Entities={training_data[0][1]['entities'][:2]}")
        return training_data
    except Exception as e:
        logger.error(f"Error loading training data: {e}", exc_info=True)
        return []

# --- NER Trainer (from nlp_model_optimized.py, with minor adaptations) ---
class NERTrainer:
    def __init__(self, train_data: List[Tuple]):
        self.train_data = train_data
        self.nlp = spacy.blank("en")
        self.metrics_history = []

    def prepare_training_examples(self) -> List[Example]:
        logger.info("Preparing training examples for NER...")
        valid_examples = []
        for text, annotations in self.train_data:
            try:
                doc = self.nlp.make_doc(text)
                example = Example.from_dict(doc, annotations) # Example.from_dict handles span alignment
                valid_examples.append(example)
            except Exception as e:
                logger.warning(f"Error creating example for text '{text[:50]}...': {e}")
        logger.info(f"Prepared {len(valid_examples)} valid training examples.")
        return valid_examples

    def train_model(self, n_iter: int = 30, dropout: float = 0.2, batch_size: int = 8):
        logger.info("Starting NER model training...")
        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe("ner")
        else:
            ner = self.nlp.get_pipe("ner")

        for _, annotations in self.train_data:
            for _, _, label in annotations.get("entities", []):
                if label not in ner.labels:
                    ner.add_label(label)

        train_examples = self.prepare_training_examples()
        if not train_examples:
            logger.error("No valid training examples found for NER.")
            return None

        # Split data for validation during training
        if len(train_examples) < 10: # Need enough data for a meaningful split
            logger.warning("Too few training examples for a train/validation split during training. Using all for training.")
            train_subset = train_examples
            val_subset = train_examples # Evaluate on training data if too small
        else:
            split_idx = int(0.8 * len(train_examples))
            train_subset = train_examples[:split_idx]
            val_subset = train_examples[split_idx:]

        logger.info(f"Training on {len(train_subset)} examples, validating on {len(val_subset)} examples.")

        self.nlp.initialize(lambda: train_subset)
        best_f1 = -1.0 # Initialize to a value lower than any possible F1
        best_model_state = None

        for epoch in range(n_iter):
            random.shuffle(train_subset)
            losses = {}
            batches = spacy.util.minibatch(train_subset, size=spacy.util.compounding(batch_size, 32.0, 1.001))
            for batch in batches:
                try:
                    self.nlp.update(batch, drop=dropout, losses=losses)
                except Exception as e:
                    logger.error(f"Error during nlp.update in epoch {epoch+1}: {e}")
                    continue # Skip problematic batch

            # Evaluate on validation set
            if val_subset:
                metrics = self.evaluate_model_performance(val_subset, f"Epoch {epoch + 1}")
                current_f1 = metrics.get('entity_level', {}).get('f1', 0.0)
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_model_state = self.nlp.to_bytes()
                    logger.info(f"New best NER model at epoch {epoch + 1} with Entity F1: {best_f1:.3f}")
            logger.info(f"Epoch {epoch + 1}/{n_iter}, NER Loss: {losses.get('ner', 0.0):.3f}")

        if best_model_state:
            self.nlp.from_bytes(best_model_state)
            logger.info(f"Restored best NER model with Entity F1: {best_f1:.3f}")

        # Final evaluation on the full test set (if provided separately) or validation set
        if val_subset: # Or use a dedicated test set if available
             final_metrics = self.evaluate_model_performance(val_subset, "Final Validation")
             self.metrics_history.append(final_metrics) # Store final metrics

        return self.nlp

    def evaluate_model_performance(self, examples: List[Example], phase: str = "Evaluation") -> Dict:
        if not examples: return {}
        logger.info(f"Running NER {phase} on {len(examples)} examples...")

        y_true_ents = []
        y_pred_ents = []

        for ex in examples:
            pred_doc = self.nlp(ex.reference.text)

            true_entities_set = set((ent.label_, ent.start_char, ent.end_char) for ent in ex.reference.ents)
            pred_entities_set = set((ent.label_, ent.start_char, ent.end_char) for ent in pred_doc.ents)

            y_true_ents.append(true_entities_set)
            y_pred_ents.append(pred_entities_set)

        all_labels = sorted(list(STANDARDIZED_LABELS)) # Use predefined labels for consistency

        per_entity_metrics = {}
        tp_total, fp_total, fn_total = 0, 0, 0

        for label in all_labels:
            tp, fp, fn = 0, 0, 0
            for true_set, pred_set in zip(y_true_ents, y_pred_ents):
                true_label_ents = {e for e in true_set if e[0] == label}
                pred_label_ents = {e for e in pred_set if e[0] == label}

                tp += len(true_label_ents.intersection(pred_label_ents))
                fp += len(pred_label_ents - true_label_ents)
                fn += len(true_label_ents - pred_label_ents)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            support = tp + fn

            per_entity_metrics[label] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': support, 'tp':tp, 'fp':fp, 'fn':fn}
            tp_total += tp
            fp_total += fp
            fn_total += fn

        overall_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        overall_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

        metrics = {
            'phase': phase,
            'total_examples': len(examples),
            'entity_level': {
                'precision': overall_precision, 'recall': overall_recall, 'f1': overall_f1,
                'true_positives': tp_total, 'false_positives': fp_total, 'false_negatives': fn_total
            },
            'per_entity': per_entity_metrics
        }
        logger.info(f"{phase} NER Results - Overall P: {overall_precision:.3f}, R: {overall_recall:.3f}, F1: {overall_f1:.3f}")
        self.metrics_history.append(metrics) # Append metrics for each evaluation phase
        return metrics

    def save_model_and_metrics(self, model_path: str, metrics_path: str):
        try:
            ensure_directory(os.path.dirname(model_path))
            self.nlp.to_disk(model_path)
            logger.info(f"NER model saved to: {model_path}")

            ensure_directory(os.path.dirname(metrics_path))
            def convert_numpy_types(obj):
                if isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
                if isinstance(obj, list): return [convert_numpy_types(i) for i in obj]
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, (np.integer, np.floating)): return obj.item()
                return obj

            serializable_metrics = convert_numpy_types(self.metrics_history)
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_metrics, f, indent=2)
            logger.info(f"NER metrics saved to: {metrics_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving NER model/metrics: {e}")
            return False

# --- Batch NLP Processor (from nlp_model_optimized.py) ---
class BatchNLPProcessor:
    def __init__(self, nlp_model, sentiment_model=None, batch_size: int = 32):
        self.nlp_model = nlp_model
        self.sentiment_model = sentiment_model
        self.batch_size = batch_size
        if self.nlp_model and "sentencizer" not in self.nlp_model.pipe_names:
            try:
                self.nlp_model.add_pipe("sentencizer", first=True)
            except Exception as e: # If already present or other issue
                 logger.warning(f"Could not add sentencizer to custom NER model: {e}. Assuming it's handled or present.")


    def process_texts_batch(self, texts: List[str], original_texts: Optional[List[str]] = None) -> List[Dict]:
        results = []
        if not texts: return results

        # Ensure original_texts are available if texts are cleaned versions
        if original_texts is None:
            original_texts = texts

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_original_texts = original_texts[i:i + self.batch_size]

            valid_indices = [idx for idx, text in enumerate(batch_texts) if text and text.strip()]
            if not valid_indices: continue

            current_valid_texts = [batch_texts[idx] for idx in valid_indices]
            current_original_texts = [batch_original_texts[idx] for idx in valid_indices]

            docs = []
            try:
                docs = list(self.nlp_model.pipe(current_valid_texts, batch_size=min(len(current_valid_texts), self.batch_size)))
            except Exception as e:
                logger.error(f"Error in NLP pipeline processing batch: {e}. Trying individual processing for this batch.")
                for text_item in current_valid_texts:
                    try:
                        docs.append(self.nlp_model(text_item))
                    except Exception as e_ind:
                        logger.error(f"Failed to process individual text '{text_item[:50]}...': {e_ind}")
                        docs.append(None) # Add placeholder for failed docs

            for doc_idx, doc in enumerate(docs):
                if doc is None: continue # Skip failed docs

                original_text_for_doc = current_original_texts[doc_idx]
                entity_counts = Counter(ent.label_ for ent in doc.ents)

                try:
                    sents = list(doc.sents)
                except ValueError: # Fallback if sentencizer fails
                    sents = [doc[:]] # Treat whole doc as one sentence

                for sent in sents:
                    sent_text = sent.text.strip()
                    if not sent_text: continue

                    sentiment_result = {"label": "unknown", "score": 0.0}
                    if self.sentiment_model:
                        try:
                            # Transformers pipeline expects a list, even for one item
                            sentiment_output = self.sentiment_model([sent_text])
                            if sentiment_output and isinstance(sentiment_output, list):
                                sentiment_result = sentiment_output[0]
                        except Exception as e:
                            logger.warning(f"Sentiment analysis failed for sentence '{sent_text[:30]}...': {e}")

                    # Find entities within the current sentence's span
                    sent_entities = [ent for ent in doc.ents if ent.start_char >= sent.start_char and ent.end_char <= sent.end_char]

                    if not sent_entities and not entity_counts: # If no entities in doc, skip sentence processing for entities
                        # Optionally, still record sentence and its sentiment if needed for other analyses
                        # results.append({ ... "Seat Component": "N/A", "Cue Word": "N/A" ...})
                        pass


                    for ent in sent_entities:
                        if ent.label_ in STANDARDIZED_LABELS:
                            results.append({
                                "Feedback Text": original_text_for_doc, # Use original for reporting
                                "Cleaned Text": doc.text, # Processed text
                                "Seat Component": ent.label_,
                                "Cue Word": ent.text,
                                "Component Frequency in Text": entity_counts.get(ent.label_, 0),
                                "Sentence Sentiment Label": sentiment_result["label"].lower(),
                                "Sentence Sentiment Score": sentiment_result["score"],
                                "Sentence Text": sent_text,
                                "Entity Start Char (Cleaned)": ent.start_char,
                                "Entity End Char (Cleaned)": ent.end_char
                            })
        return results

# --- Model Manager (from nlp_model_optimized.py) ---
class ModelManager:
    def __init__(self, custom_ner_model_path: Optional[str] = "ultimate_nlp_output/models/ultimate_ner_model"):
        """
        Initialize ModelManager with ultimate NER model as default.
        
        Args:
            custom_ner_model_path: Path to the ultimate custom NER model (achieved 100% targets)
        """
        self.custom_ner_model_path = custom_ner_model_path
        self._nlp_model = None
        self._sentiment_model = None
        self._nlp_core_for_lda = None

    @property
    def nlp_model(self): # This is the custom trained NER model
        if self._nlp_model is None:
            if self.custom_ner_model_path and os.path.exists(self.custom_ner_model_path):
                try:
                    self._nlp_model = spacy.load(self.custom_ner_model_path)
                    logger.info(f"Loaded custom NER model from {self.custom_ner_model_path}")
                except Exception as e:
                    logger.error(f"Failed to load custom NER model from {self.custom_ner_model_path}: {e}. Using blank model.")
                    self._nlp_model = spacy.blank("en")
            else:
                logger.warning(f"Custom NER model path '{self.custom_ner_model_path}' not found. Using blank model for NER.")
                self._nlp_model = spacy.blank("en")
        return self._nlp_model

    @property
    def sentiment_model(self):
        if self._sentiment_model is None:
            try:
                self._sentiment_model = pipeline(
                    "sentiment-analysis", model="siebert/sentiment-roberta-large-english",
                    truncation=True, max_length=512, device=0 if torch.cuda.is_available() else -1, batch_size=8
                )
                logger.info("Loaded sentiment model: siebert/sentiment-roberta-large-english")
            except Exception as e:
                logger.warning(f"Failed to load preferred sentiment model (siebert/sentiment-roberta-large-english): {e}. Trying default.")
                try:
                    self._sentiment_model = pipeline(
                        "sentiment-analysis", truncation=True, max_length=512,
                        device=-1, batch_size=4 # Force CPU for fallback
                    )
                    logger.info("Loaded fallback default sentiment model.")
                except Exception as e2:
                    logger.error(f"Failed to load any sentiment model: {e2}")
                    self._sentiment_model = None
        return self._sentiment_model

    @property
    def nlp_core_for_lda(self): # For LDA preprocessing, avoid interfering with custom NER model
        if self._nlp_core_for_lda is None:
            try:
                # Load a base model without NER to avoid conflicts if custom NER is also loaded
                self._nlp_core_for_lda = spacy.load("en_core_web_sm", disable=["ner", "parser"])
                logger.info("Loaded en_core_web_sm (for LDA preprocessing) with NER and parser disabled.")
            except OSError:
                logger.warning("en_core_web_sm not found. LDA preprocessing might be less effective. Trying blank model for LDA.")
                self._nlp_core_for_lda = spacy.blank("en")
                if "sentencizer" not in self._nlp_core_for_lda.pipe_names:
                     self._nlp_core_for_lda.add_pipe("sentencizer")

            except Exception as e:
                logger.error(f"Failed to load en_core_web_sm for LDA: {e}. Using blank model.")
                self._nlp_core_for_lda = spacy.blank("en")
                if "sentencizer" not in self._nlp_core_for_lda.pipe_names:
                     self._nlp_core_for_lda.add_pipe("sentencizer")
        return self._nlp_core_for_lda


# --- Kansei Engineering Module (Adapted from train-seat-nlp-analysis-v2.py) ---
class KanseiModule:
    def __init__(self, lda_model, dictionary, corpus, full_df_with_ner_sentiment):
        self.lda_model = lda_model
        self.dictionary = dictionary
        self.corpus = corpus # This corpus is based on preprocessed texts for LDA
        self.df = full_df_with_ner_sentiment # This df has original text, NER entities, sentiment
        self.stop_words_kansei = set(nltk.corpus.stopwords.words('english')) # Specific stopwords for Kansei text processing

    def _preprocess_text_for_kansei_lda(self, text):
        """Preprocessing specific for feeding text to the trained LDA model for Kansei mapping."""
        if pd.isna(text) or text == '': return []
        text = str(text).strip().lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = nltk.tokenize.word_tokenize(text)
        # Using the same compound word logic as original Kansei script
        compound_words = {
            'head rest': 'headrest', 'arm rest': 'armrest', 'back rest': 'backrest',
            'leg room': 'legroom', 'seat cushion': 'seatcushion'
        }
        processed_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1:
                compound = f"{tokens[i]} {tokens[i+1]}"
                if compound in compound_words:
                    processed_tokens.append(compound_words[compound])
                    i += 2
                    continue
            processed_tokens.append(tokens[i])
            i += 1

        return [word for word in processed_tokens if word not in self.stop_words_kansei and len(word) > 2 and not word.isdigit()]

    def map_topics_to_kansei(self):
        logger.info("Mapping LDA topics to Kansei emotions...")
        kansei_mapping = { # From train-seat-nlp-analysis-v2.py
            0: {"primary_emotion": "Uncomfortable", "secondary_emotions": ["Painful", "Stiff", "Harsh"], "keywords": ["hard", "keras", "sakit", "pain", "stiff", "uncomfortable", "hurt", "ache", "rigid", "firm"]},
            1: {"primary_emotion": "Cramped", "secondary_emotions": ["Confined", "Restricted", "Tight"], "keywords": ["narrow", "sempit", "tight", "small", "kecil", "cramped", "confined", "restricted", "squeezed"]},
            2: {"primary_emotion": "Comfortable", "secondary_emotions": ["Cozy", "Relaxing", "Pleasant"], "keywords": ["comfortable", "nyaman", "good", "soft", "empuk", "cozy", "relaxing", "pleasant", "soothing"]},
            3: {"primary_emotion": "Spacious", "secondary_emotions": ["Roomy", "Open", "Airy"], "keywords": ["spacious", "luas", "wide", "room", "space", "roomy", "open", "airy", "expansive"]},
            4: {"primary_emotion": "Premium", "secondary_emotions": ["Luxurious", "Elegant", "Sophisticated"], "keywords": ["premium", "quality", "luxury", "bagus", "best", "luxurious", "elegant", "sophisticated"]},
            5: {"primary_emotion": "Supportive", "secondary_emotions": ["Ergonomic", "Stable", "Secure"], "keywords": ["support", "position", "posisi", "ergonomic", "stable", "secure", "balanced", "aligned"]},
            6: {"primary_emotion": "Innovative", "secondary_emotions": ["Modern", "Advanced", "Smart"], "keywords": ["innovative", "modern", "advanced", "cutting-edge", "futuristic", "smart", "high-tech"]},
            7: {"primary_emotion": "Disappointing", "secondary_emotions": ["Inadequate", "Subpar", "Poor"], "keywords": ["disappointing", "inadequate", "subpar", "mediocre", "unsatisfactory", "poor", "lacking"]},
            8: {"primary_emotion": "Relaxing", "secondary_emotions": ["Calming", "Therapeutic", "Restorative"], "keywords": ["relaxing", "calming", "stress-free", "therapeutic", "rejuvenating", "restorative", "healing"]},
            9: {"primary_emotion": "Exciting", "secondary_emotions": ["Thrilling", "Stimulating", "Dynamic"], "keywords": ["exciting", "thrilling", "stimulating", "energizing", "dynamic", "vibrant", "invigorating"]}
        }

        kansei_results = []
        # Iterate through the main DataFrame which has NER and sentiment results
        # For each original review text, get its LDA topic distribution

        # We need to map original reviews to their LDA representation if corpus was built on different preprocessed texts
        # Assuming self.df has 'Cleaned Text' used for NER/Sentiment, and LDA was on similar.
        # If LDA preprocessing was different, this mapping needs care.
        # For now, assume self.corpus corresponds to a list of texts that can be mapped back to self.df

        # Create a mapping from the original review text (or a unique ID) to its LDA topic distribution
        # The `self.corpus` was created from `lda_documents` in `generate_enhanced_lda_analysis`.
        # We need to link `self.df` rows to these `lda_documents`.
        # Let's assume `generate_enhanced_lda_analysis` returns the texts it used for LDA.
        # For this integration, we'll re-calculate topic distribution for each review text in self.df

        for idx, row in self.df.iterrows():
            # Use the 'Cleaned Text' that was processed by NER/Sentiment for consistency,
            # but preprocess it again specifically for the Kansei LDA model's dictionary.
            review_text_for_lda = str(row.get('Cleaned Text', '')).strip() # Text used for NER/Sentiment
            if not review_text_for_lda: continue

            processed_for_lda = self._preprocess_text_for_kansei_lda(review_text_for_lda)
            if not processed_for_lda:
                doc_topics = []
            else:
                bow = self.dictionary.doc2bow(processed_for_lda)
                if not bow: # If text results in empty BoW for LDA dictionary
                    doc_topics = []
                else:
                    doc_topics = self.lda_model.get_document_topics(bow, minimum_probability=0.01)

            primary_emotion = "Unknown"
            secondary_emotions = []
            emotion_confidence = 0.0
            dominant_topic_idx = None

            if doc_topics:
                dominant_topic = max(doc_topics, key=lambda x: x[1])
                dominant_topic_idx = dominant_topic[0]
                emotion_confidence = dominant_topic[1]
                if dominant_topic_idx in kansei_mapping:
                    primary_emotion = kansei_mapping[dominant_topic_idx]["primary_emotion"]
                    secondary_emotions = kansei_mapping[dominant_topic_idx]["secondary_emotions"]

                # Keyword override logic from Kansei script
                original_review_lower = str(row.get('Feedback Text', '')).lower() # Use original for keyword matching
                keyword_emotions = []
                for topic_id_map, emotion_data in kansei_mapping.items():
                    keyword_matches = sum(1 for keyword in emotion_data["keywords"] if keyword in original_review_lower)
                    if keyword_matches > 0:
                        keyword_emotions.append({
                            'emotion': emotion_data["primary_emotion"],
                            'matches': keyword_matches,
                            'secondary': emotion_data["secondary_emotions"]
                        })
                if keyword_emotions:
                    best_keyword_emotion = max(keyword_emotions, key=lambda x: x['matches'])
                    # Threshold for keyword override (e.g., if keyword match is strong)
                    # Or if LDA confidence is low and keyword match is present
                    if best_keyword_emotion['matches'] >= 2 or (emotion_confidence < 0.5 and best_keyword_emotion['matches'] > 0) :
                        primary_emotion = best_keyword_emotion['emotion']
                        secondary_emotions = best_keyword_emotion['secondary']
                        # Optionally update confidence if overridden by keywords
                        # emotion_confidence = max(emotion_confidence, 0.5) # Arbitrary confidence boost

            # Collect all entities and sentiments for this specific review_id/text
            # The self.df is already structured with one row per entity-sentence mention.
            # We need to aggregate this back to a per-review level for Kansei.
            # This current loop is per row in self.df. We need to associate this Kansei emotion
            # with all rows in self.df that correspond to the same original "Feedback Text".

            # For now, this function will return a list of emotions per original review.
            # The calling function will need to merge this back.
            kansei_results.append({
                'Feedback Text': row.get('Feedback Text'), # Link back to original review
                'dominant_lda_topic': dominant_topic_idx,
                'kansei_emotion': primary_emotion,
                'kansei_secondary_emotions': secondary_emotions,
                'kansei_emotion_confidence': emotion_confidence,
                # 'Seat Component': row.get('Seat Component'), # This will be added when merging
                # 'Cue Word': row.get('Cue Word'),
                # 'Sentence Sentiment Label': row.get('Sentence Sentiment Label')
            })

        # Deduplicate based on 'Feedback Text' as Kansei emotion is per review
        deduplicated_kansei_results = []
        seen_texts = set()
        for res in kansei_results:
            if res['Feedback Text'] not in seen_texts:
                deduplicated_kansei_results.append(res)
                seen_texts.add(res['Feedback Text'])

        logger.info(f"Mapped {len(deduplicated_kansei_results)} reviews to Kansei emotions.")
        return deduplicated_kansei_results


    def analyze_emotion_patterns(self, kansei_results_per_review: List[Dict]):
        logger.info("Analyzing Kansei emotion patterns...")
        # This function is from train-seat-nlp-analysis-v2.py, largely unchanged
        # It expects kansei_results to be one entry per review with its Kansei emotion.
        emotion_analysis = {
            'primary_emotions': Counter([r['kansei_emotion'] for r in kansei_results_per_review]),
            'secondary_emotions': Counter(),
            'emotion_combinations': defaultdict(int),
            'confidence_levels': defaultdict(list),
            # 'source_emotion_patterns': defaultdict(lambda: defaultdict(int)) # Source not in current df
        }
        for result in kansei_results_per_review:
            primary = result['kansei_emotion']
            secondary = result.get('kansei_secondary_emotions', [])
            confidence = result.get('kansei_emotion_confidence', 0.0)
            # source = result.get('Source', 'Unknown') # Source column not in current combined df

            for sec_emotion in secondary:
                emotion_analysis['secondary_emotions'][sec_emotion] += 1
            if secondary:
                combo = f"{primary} + {', '.join(secondary[:2])}"
                emotion_analysis['emotion_combinations'][combo] += 1
            emotion_analysis['confidence_levels'][primary].append(confidence)
            # emotion_analysis['source_emotion_patterns'][source][primary] += 1

        emotion_analysis['avg_confidence'] = {
            emotion: sum(confs) / len(confs) if confs else 0.0
            for emotion, confs in emotion_analysis['confidence_levels'].items()
        }
        return emotion_analysis

    def analyze_emotion_trends(self, emotion_patterns: Dict):
        logger.info("Analyzing Kansei emotion trends...")
        # This function is from train-seat-nlp-analysis-v2.py, largely unchanged
        trends = {'dominant_emotions': [], 'emerging_emotions': [], 'emotion_intensity': {}}
        total_emotions = sum(emotion_patterns['primary_emotions'].values())
        if total_emotions == 0: return trends # Avoid division by zero

        for emotion, count in emotion_patterns['primary_emotions'].items():
            percentage = (count / total_emotions) * 100
            if percentage > 15:
                trends['dominant_emotions'].append({
                    'emotion': emotion, 'percentage': percentage,
                    'confidence': emotion_patterns['avg_confidence'].get(emotion, 0.0)
                })
        for emotion, count in emotion_patterns['secondary_emotions'].most_common(5):
            if count > total_emotions * 0.05:
                trends['emerging_emotions'].append({
                    'emotion': emotion, 'frequency': count,
                    'potential': 'High' if count > total_emotions * 0.1 else 'Medium'
                })
        for emotion, confidences in emotion_patterns['confidence_levels'].items():
            if confidences:
                avg_conf = sum(confidences) / len(confidences)
                std_conf = np.std(confidences) if len(confidences) > 1 else 0.0
                trends['emotion_intensity'][emotion] = {
                    'average': avg_conf,
                    'consistency': 'High' if std_conf < 0.2 else 'Medium' if std_conf < 0.4 else 'Low'
                }
        return trends

    def generate_design_insights(self, kansei_results_per_review: List[Dict], processed_ner_sentiment_df: pd.DataFrame):
        logger.info("Generating Kansei design insights...")
        emotion_patterns = self.analyze_emotion_patterns(kansei_results_per_review)

        # Merge Kansei emotions back to the main processed_ner_sentiment_df
        kansei_df = pd.DataFrame(kansei_results_per_review)
        # Ensure 'Feedback Text' is the common key
        merged_df_for_insights = pd.merge(processed_ner_sentiment_df, kansei_df, on='Feedback Text', how='left')

        feature_emotion_map = defaultdict(lambda: defaultdict(int))
        if not merged_df_for_insights.empty:
             for _, row in merged_df_for_insights.iterrows():
                feat_type = row.get('Seat Component')
                emotion = row.get('kansei_emotion')
                if feat_type and emotion and emotion != "Unknown": # Only count if component and emotion are known
                    feature_emotion_map[feat_type][emotion] += 1

        insights = {
            'overall_kansei_sentiment': {},
            'kansei_secondary_emotions': dict(emotion_patterns['secondary_emotions']),
            'kansei_emotion_combinations': dict(emotion_patterns['emotion_combinations']),
            'kansei_emotion_confidence': emotion_patterns['avg_confidence'],
            'feature_kansei_emotion_correlation': dict(feature_emotion_map),
            'design_recommendations': self.generate_recommendations(kansei_results_per_review), # Pass per-review results
            'kansei_emotion_trends': self.analyze_emotion_trends(emotion_patterns)
        }

        all_emotions = [r['kansei_emotion'] for r in kansei_results_per_review if r['kansei_emotion'] != "Unknown"]
        emotion_counts = Counter(all_emotions)
        total_valid_emotions = sum(emotion_counts.values())

        for emotion, count in emotion_counts.items():
            if total_valid_emotions > 0:
                confidence = emotion_patterns['avg_confidence'].get(emotion, 0.0)
                insights['overall_kansei_sentiment'][emotion] = {
                    'count': count,
                    'percentage': (count / total_valid_emotions) * 100 if total_valid_emotions else 0,
                    'avg_confidence': confidence,
                    'reliability': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.4 else 'Low'
                }
        return insights, merged_df_for_insights # Return merged_df for saving

    def generate_recommendations(self, kansei_results_per_review: List[Dict]):
        logger.info("Generating Kansei design recommendations...")
        # This function is from train-seat-nlp-analysis-v2.py, largely unchanged
        # It expects kansei_results to be one entry per review.
        emotion_counts = Counter([r['kansei_emotion'] for r in kansei_results_per_review if r['kansei_emotion'] != "Unknown"])
        total_reviews_with_emotion = sum(emotion_counts.values())
        if total_reviews_with_emotion == 0: return []

        recommendations = []
        # Comfort-related
        discomfort_ratio = (emotion_counts.get('Uncomfortable', 0) + emotion_counts.get('Disappointing', 0)) / total_reviews_with_emotion
        if discomfort_ratio > 0.2: # Example threshold
            recommendations.append({'component': 'Seat Comfort System', 'issue': f'High discomfort ({discomfort_ratio:.1%})', 'priority': 'High', 'suggestions': ['Improve cushioning', 'Add adjustable lumbar support']})
        # Space-related
        cramped_ratio = emotion_counts.get('Cramped', 0) / total_reviews_with_emotion
        if cramped_ratio > 0.15:
            recommendations.append({'component': 'Space Optimization', 'issue': f'Significant cramped feeling ({cramped_ratio:.1%})', 'priority': 'High', 'suggestions': ['Increase legroom', 'Optimize seat width']})
        # Premium experience
        premium_ratio = emotion_counts.get('Premium', 0) / total_reviews_with_emotion
        if premium_ratio < 0.2:
             recommendations.append({'component': 'Premium Experience', 'issue': f'Low premium perception ({premium_ratio:.1%})', 'priority': 'Medium', 'suggestions': ['Upgrade materials', 'Add luxury features like massage']})
        # Add more based on other Kansei emotions...
        # Supportive
        supportive_ratio = emotion_counts.get('Supportive', 0) / total_reviews_with_emotion
        if supportive_ratio < 0.25 and emotion_counts.get('Uncomfortable',0) > 0 : # If not supportive and some discomfort
            recommendations.append({'component': 'Ergonomic Support', 'issue': f'Insufficient support ({supportive_ratio:.1%}), linked to discomfort', 'priority': 'Medium', 'suggestions': ['Enhance backrest contour', 'Improve headrest adjustability']})

        priority_order = {'High': 1, 'Medium': 2, 'Low': 3}
        recommendations.sort(key=lambda x: priority_order.get(x.get('priority', 'Low'), 3))
        return recommendations

# --- LDA Analysis (Adapted from nlp_model_optimized.py and Kansei script) ---
def preprocess_text_for_lda(text: str, nlp_core_model, stop_words_set: set) -> List[str]:
    """Preprocessing for LDA using spaCy for lemmatization and POS tagging."""
    if pd.isna(text) or not text.strip():
        return []

    # Basic cleaning first (consistent with clean_text_cached but without lowercasing yet for POS)
    text = BeautifulSoup(str(text), "lxml").get_text(" ")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)

    doc = nlp_core_model(text) # Use the passed nlp_core_model
    tokens = [
        token.lemma_.lower() for token in doc
        if token.is_alpha and
        not token.is_stop and # spaCy's stop words
        token.lemma_.lower() not in stop_words_set and # Custom stop words
        len(token.lemma_) > 2 and
        token.pos_ in ['NOUN', 'ADJ', 'VERB'] # Focus on content words
    ]
    return tokens

def generate_combined_lda_analysis(
    full_df: pd.DataFrame,
    text_column_for_lda: str, # e.g., 'clean_text' or 'Feedback Text'
    nlp_core_model, # Pass the loaded en_core_web_sm (or blank)
    num_topics: int = 10,
    output_dir: str = "combined_nlp_output/lda_analysis"
    ):
    logger.info("Starting combined LDA analysis...")
    ensure_directory(output_dir)

    feedback_data_for_lda = full_df[text_column_for_lda].dropna().astype(str).tolist()
    if not feedback_data_for_lda:
        logger.warning("No text data available for LDA analysis.")
        return None, None, None

    # Define stopwords for LDA
    custom_stop_words_lda = set(nltk.corpus.stopwords.words("english")) | set(string.punctuation) | {
        "seat", "seats", "car", "vehicle", "trip", "feature", "get", "feel", "felt", "look", "make", "also",
        "even", "really", "quite", "very", "much", "good", "great", "nice", "well", "drive", "driving",
        "would", "could", "im", "ive", "id", "nan", "auto", "automobile", "product", "item", "order"
    }

    processed_texts_for_lda = [
        preprocess_text_for_lda(text, nlp_core_model, custom_stop_words_lda)
        for text in feedback_data_for_lda
    ]
    processed_texts_for_lda = [text for text in processed_texts_for_lda if len(text) > 1] # Need at least 2 tokens for LDA

    if len(processed_texts_for_lda) < 5: # Need a minimum number of documents
        logger.warning("Insufficient processed documents for meaningful LDA analysis.")
        return None, None, None

    dictionary = corpora.Dictionary(processed_texts_for_lda)
    dictionary.filter_extremes(no_below=3, no_above=0.6, keep_n=10000) # Adjusted parameters

    if not dictionary:
        logger.warning("LDA dictionary is empty after filtering. Cannot proceed with LDA.")
        return None, None, None

    corpus = [dictionary.doc2bow(text) for text in processed_texts_for_lda]
    corpus = [doc for doc in corpus if doc] # Remove empty documents from corpus

    if not corpus or len(corpus) < num_topics:
        logger.warning(f"Corpus has too few documents ({len(corpus)}) for {num_topics} topics. Adjusting num_topics or cannot proceed.")
        if not corpus: return None, None, None
        num_topics = max(1, len(corpus)-1) if len(corpus) > 1 else 1 # Ensure num_topics is valid
        if num_topics == 0: return None, None, None
        logger.info(f"Adjusted num_topics to {num_topics}")


    logger.info(f"Training LDA model with {num_topics} topics on {len(corpus)} documents...")
    lda_model = LdaModel(
        corpus=corpus, id2word=dictionary, num_topics=num_topics,
        random_state=42, update_every=1, chunksize=100,
        passes=15, alpha='auto', per_word_topics=True, iterations=100
    )

    # Save LDA model, dictionary, corpus (needed by KanseiModule)
    lda_model.save(os.path.join(output_dir, "lda_model.gensim"))
    dictionary.save(os.path.join(output_dir, "lda_dictionary.gensim"))
    corpora.MmCorpus.serialize(os.path.join(output_dir, "lda_corpus.mm"), corpus)

    # Save processed texts used for LDA for coherence calculation if needed later
    with open(os.path.join(output_dir, "lda_processed_texts.json"), 'w') as f:
        json.dump(processed_texts_for_lda, f)

    logger.info("LDA model, dictionary, and corpus saved.")

    # Coherence Model
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_texts_for_lda, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    logger.info(f"LDA Coherence Score (c_v): {coherence_lda:.4f}")

    # Visualization (pyLDAvis)
    try:
        vis_data = gensimvis.prepare(lda_model, corpus, dictionary, mds='mmds') # Use mmds for potentially better layout
        pyLDAvis.save_html(vis_data, os.path.join(output_dir, 'lda_visualization.html'))
        logger.info(f"LDA pyLDAvis visualization saved to {output_dir}/lda_visualization.html")
    except Exception as e:
        logger.error(f"Error generating pyLDAvis visualization: {e}", exc_info=True)
        logger.warning("Ensure you have a C++ compiler for pyLDAvis if using certain mds methods, or try mds='tsne'.")


    # Save topic details
    lda_topics_data = []
    for idx, topic in lda_model.print_topics(-1, num_words=15):
        lda_topics_data.append({"topic_id": idx, "terms": topic})
        logger.info(f"Topic {idx}: {topic}")

    pd.DataFrame(lda_topics_data).to_csv(os.path.join(output_dir, "lda_topic_terms.csv"), index=False)

    return lda_model, dictionary, corpus # Return these for Kansei module


# --- Visualization and Reporting (Combined and Enhanced) ---
def generate_ner_metrics_visualization(metrics_history: List[Dict], output_dir: str):
    """
    Generate comprehensive NER metrics visualizations including:
    1. Overall entity-level performance over training phases
    2. Per-entity performance over training phases (separate plot for each entity)
    3. Final per-entity performance comparison
    """
    if not metrics_history:
        logger.warning("No NER metrics history to visualize.")
        return

    ensure_directory(os.path.join(output_dir, "plots"))
    ensure_directory(os.path.join(output_dir, "plots", "per_entity"))
    epochs = [i + 1 for i in range(len(metrics_history))]

    # Extract last (final) metrics for entity-level P, R, F1
    final_eval_metrics = None
    for m in reversed(metrics_history):
        if 'entity_level' in m:
            final_eval_metrics = m
            break

    if not final_eval_metrics:
        logger.warning("Could not find final NER evaluation metrics in history for plotting.")
        return

    # 1. Overall entity-level performance over training phases
    entity_f1s = [m.get('entity_level', {}).get('f1', 0) for m in metrics_history if 'entity_level' in m]
    entity_ps = [m.get('entity_level', {}).get('precision', 0) for m in metrics_history if 'entity_level' in m]
    entity_rs = [m.get('entity_level', {}).get('recall', 0) for m in metrics_history if 'entity_level' in m]

    # Ensure epochs list matches the length of extracted metrics
    valid_epochs = epochs[:len(entity_f1s)]

    plt.figure(figsize=(12, 6))
    plt.plot(valid_epochs, entity_ps, 'b-o', label='Precision (Entity)', linewidth=2, markersize=6)
    plt.plot(valid_epochs, entity_rs, 'r-s', label='Recall (Entity)', linewidth=2, markersize=6)
    plt.plot(valid_epochs, entity_f1s, 'g-^', label='F1-Score (Entity)', linewidth=2, markersize=6)
    plt.title('NER Overall Entity-Level Performance Over Training/Evaluation Phases', fontsize=14, fontweight='bold')
    plt.xlabel('Evaluation Phase/Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(valid_epochs)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "ner_training_performance.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Per-entity performance over training phases (separate plot for each entity)
    # Collect all unique entity labels from all metrics
    all_entities = set()
    for m in metrics_history:
        if 'per_entity' in m and m['per_entity']:
            all_entities.update(m['per_entity'].keys())
    
    all_entities = sorted(list(all_entities))
    logger.info(f"Creating individual performance plots for {len(all_entities)} entities: {all_entities}")

    # Extract per-entity metrics over time
    per_entity_over_time = {}
    for entity in all_entities:
        per_entity_over_time[entity] = {
            'precision': [],
            'recall': [],
            'f1': [],
            'epochs': []
        }
    
    # Collect metrics for each epoch
    for epoch_idx, m in enumerate(metrics_history):
        if 'per_entity' in m and m['per_entity']:
            for entity in all_entities:
                if entity in m['per_entity']:
                    metrics = m['per_entity'][entity]
                    per_entity_over_time[entity]['precision'].append(metrics.get('precision', 0))
                    per_entity_over_time[entity]['recall'].append(metrics.get('recall', 0))
                    per_entity_over_time[entity]['f1'].append(metrics.get('f1', 0))
                    per_entity_over_time[entity]['epochs'].append(epoch_idx + 1)
                else:
                    # Entity not found in this epoch (likely no examples)
                    per_entity_over_time[entity]['precision'].append(0)
                    per_entity_over_time[entity]['recall'].append(0)
                    per_entity_over_time[entity]['f1'].append(0)
                    per_entity_over_time[entity]['epochs'].append(epoch_idx + 1)

    # Create individual plots for each entity
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for entity in all_entities:
        if not per_entity_over_time[entity]['epochs']:
            continue
            
        plt.figure(figsize=(10, 6))
        
        epochs_entity = per_entity_over_time[entity]['epochs']
        precision_entity = per_entity_over_time[entity]['precision']
        recall_entity = per_entity_over_time[entity]['recall']
        f1_entity = per_entity_over_time[entity]['f1']
        
        plt.plot(epochs_entity, precision_entity, 'b-o', label='Precision', linewidth=2.5, markersize=7)
        plt.plot(epochs_entity, recall_entity, 'r-s', label='Recall', linewidth=2.5, markersize=7)
        plt.plot(epochs_entity, f1_entity, 'g-^', label='F1-Score', linewidth=2.5, markersize=7)
        
        plt.title(f'NER Performance Over Training: {entity}', fontsize=14, fontweight='bold')
        plt.xlabel('Evaluation Phase/Epoch', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(epochs_entity)
        plt.ylim(0, 1.05)
        
        # Add final score annotations
        if f1_entity:
            final_f1 = f1_entity[-1]
            final_precision = precision_entity[-1]
            final_recall = recall_entity[-1]
            
            plt.text(0.02, 0.98, f'Final Scores:\nF1: {final_f1:.3f}\nP: {final_precision:.3f}\nR: {final_recall:.3f}', 
                    transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save with entity name in filename
        safe_entity_name = entity.replace('/', '_').replace(' ', '_')
        plt.savefig(os.path.join(output_dir, "plots", "per_entity", f"ner_performance_{safe_entity_name}.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()

    # 3. Final per-entity performance comparison (existing bar chart)
    per_entity_metrics = final_eval_metrics.get('per_entity', {})
    if per_entity_metrics:
        labels = list(per_entity_metrics.keys())
        f1_scores = [per_entity_metrics[l]['f1'] for l in labels]
        precision_scores = [per_entity_metrics[l]['precision'] for l in labels]
        recall_scores = [per_entity_metrics[l]['recall'] for l in labels]

        df_per_label = pd.DataFrame({
            'Label': labels,
            'Precision': precision_scores,
            'Recall': recall_scores,
            'F1-Score': f1_scores
        }).set_index('Label')

        ax = df_per_label.plot(kind='bar', figsize=(14, 7), colormap='viridis', width=0.8)
        plt.title('Final NER Performance per Entity Label', fontsize=14, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.xlabel('Entity Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.05)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=8, rotation=90, padding=3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plots", "ner_per_label_performance.png"), dpi=150, bbox_inches='tight')
        plt.close()

    # 4. Create a summary grid plot showing all entities together
    if len(all_entities) > 0:
        # Calculate grid dimensions
        n_entities = len(all_entities)
        n_cols = min(3, n_entities)  # Max 3 columns
        n_rows = (n_entities + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_entities == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_entities == 1 else axes
        else:
            axes = axes.flatten()
        
        for idx, entity in enumerate(all_entities):
            ax = axes[idx] if n_entities > 1 else axes[0]
            
            epochs_entity = per_entity_over_time[entity]['epochs']
            f1_entity = per_entity_over_time[entity]['f1']
            
            if epochs_entity and f1_entity:
                ax.plot(epochs_entity, f1_entity, 'g-o', linewidth=2, markersize=5)
                ax.set_title(f'{entity}', fontsize=11, fontweight='bold')
                ax.set_xlabel('Epoch', fontsize=9)
                ax.set_ylabel('F1-Score', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1.05)
                
                # Add final F1 score as text
                if f1_entity:
                    final_f1 = f1_entity[-1]
                    ax.text(0.95, 0.95, f'F1: {final_f1:.3f}', transform=ax.transAxes, 
                           fontsize=9, ha='right', va='top',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        
        # Hide unused subplots
        for idx in range(n_entities, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('NER F1-Score Progress for All Entities', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plots", "ner_all_entities_grid.png"), dpi=150, bbox_inches='tight')
        plt.close()

    logger.info(f"NER metrics visualizations saved:")
    logger.info(f"  - Overall performance: plots/ner_training_performance.png")
    logger.info(f"  - Individual entity plots: plots/per_entity/ ({len(all_entities)} files)")
    logger.info(f"  - Final comparison: plots/ner_per_label_performance.png")
    logger.info(f"  - Grid summary: plots/ner_all_entities_grid.png")


def generate_kansei_visualizations(kansei_insights: Dict, output_dir: str):
    logger.info("Generating Kansei visualizations...")
    ensure_directory(os.path.join(output_dir, "plots"))

    # 1. Kansei Emotion Distribution (Primary)
    primary_emotions = kansei_insights.get('overall_kansei_sentiment', {})
    if primary_emotions:
        labels = [data.get('emotion', emotion_name) for emotion_name, data in primary_emotions.items()] # Use emotion_name if 'emotion' key missing
        counts = [data['count'] for data in primary_emotions.values()]
        percentages = [data['percentage'] for data in primary_emotions.values()]

        # Filter out 'Unknown' or zero-count emotions for cleaner plot
        filtered_labels, filtered_counts, filtered_percentages = [], [], []
        for l, c, p in zip(labels, counts, percentages):
            if l.lower() != 'unknown' and c > 0 :
                 filtered_labels.append(l)
                 filtered_counts.append(c)
                 filtered_percentages.append(p)


        if filtered_labels:
            plt.figure(figsize=(12, 7))
            bars = plt.bar(filtered_labels, filtered_counts, color=sns.color_palette("pastel", len(filtered_labels)))
            plt.title('Kansei Primary Emotion Distribution', fontsize=16)
            plt.xlabel('Kansei Emotion', fontsize=12)
            plt.ylabel('Number of Reviews', fontsize=12)
            plt.xticks(rotation=45, ha="right", fontsize=10)
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}\n({filtered_percentages[i]:.1f}%)',
                         ha='center', va='bottom', fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "plots", "kansei_emotion_distribution.png"), dpi=150)
            plt.close()

    # 2. Feature-Kansei Emotion Correlation (Heatmap or Stacked Bar)
    feature_emotion_corr = kansei_insights.get('feature_kansei_emotion_correlation', {})
    if feature_emotion_corr:
        df_corr = pd.DataFrame(feature_emotion_corr).fillna(0).astype(int)
        # Filter out components with no emotions or emotions with no components
        df_corr = df_corr.loc[(df_corr.sum(axis=1) != 0), (df_corr.sum(axis=0) != 0)]

        if not df_corr.empty:
            plt.figure(figsize=(14, 8))
            sns.heatmap(df_corr, annot=True, fmt="d", cmap="YlGnBu", linewidths=.5)
            plt.title('Seat Component vs. Kansei Emotion Frequency', fontsize=16)
            plt.xlabel('Kansei Emotion', fontsize=12)
            plt.ylabel('Seat Component', fontsize=12)
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "plots", "feature_kansei_correlation_heatmap.png"), dpi=150)
            plt.close()
    logger.info("Kansei visualizations saved.")


def generate_general_visualizations(processed_df: pd.DataFrame, output_dir: str):
    logger.info("Generating general NLP visualizations...")
    ensure_directory(os.path.join(output_dir, "plots"))
    ensure_directory(os.path.join(output_dir, "tables"))

    if processed_df.empty:
        logger.warning("Processed DataFrame is empty, skipping general visualizations.")
        return

    # 1. Sentiment Distribution by Component (from nlp_model_optimized)
    if 'Seat Component' in processed_df.columns and 'Sentence Sentiment Label' in processed_df.columns:
        sent_pivot = processed_df.groupby(["Seat Component", "Sentence Sentiment Label"]).size().unstack(fill_value=0)
        if not sent_pivot.empty:
            sent_pivot.plot(kind="bar", stacked=True, figsize=(14, 7), colormap="viridis")
            plt.title("Transformer Sentiment Distribution by Seat Component", fontsize=16)
            plt.xlabel("Seat Component", fontsize=12)
            plt.ylabel("Number of Sentences", fontsize=12)
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.legend(title="Sentiment Label")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "plots", "component_transformer_sentiment.png"), dpi=150)
            plt.close()
            sent_pivot.to_csv(os.path.join(output_dir, "tables", "component_transformer_sentiment_table.csv"))

    # 2. Component Mention Frequency
    if 'Seat Component' in processed_df.columns:
        component_counts = processed_df['Seat Component'].value_counts()
        if not component_counts.empty:
            plt.figure(figsize=(12, 6))
            component_counts.plot(kind='bar', color=sns.color_palette("Spectral", len(component_counts)))
            plt.title("Seat Component Mention Frequency (NER)", fontsize=16)
            plt.xlabel("Seat Component", fontsize=12)
            plt.ylabel("Frequency of Mentions", fontsize=12)
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "plots", "ner_component_frequency.png"), dpi=150)
            plt.close()
            component_counts.to_csv(os.path.join(output_dir, "tables", "ner_component_frequency_table.csv"))

    # 3. Word Clouds
    if 'Cleaned Text' in processed_df.columns:
        all_cleaned_text = " ".join(processed_df['Cleaned Text'].dropna().astype(str))
        if all_cleaned_text.strip():
            stop_words_wc = set(WC_STOPWORDS) | set(nltk.corpus.stopwords.words('english')) | \
                            set(string.punctuation) | \
                            {'seat', 'seats', 'car', 'vehicle', 'also', 'get', 'got', 'would', 'could', 'make', 'made', 'see', 'really', 'even', 'one', 'nan', 'lot', 'bit', 'im', 'ive', 'id', 'well', 'good', 'great', 'nice', 'bad', 'poor', 'drive', 'driving', 'ride', 'riding', 'trip', 'product', 'item'}

            ensure_directory(os.path.join(output_dir, "wordclouds"))
            try:
                wordcloud = WordCloud(width=1200, height=600, background_color='white', stopwords=stop_words_wc, collocations=False).generate(all_cleaned_text)
                plt.figure(figsize=(12,6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.title("Overall Word Cloud (Cleaned Text)", fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "wordclouds", "overall_wordcloud.png"), dpi=150)
                plt.close()
            except Exception as e:
                 logger.error(f"Could not generate overall wordcloud: {e}")
    logger.info("General visualizations saved.")


def create_combined_report(
    processed_df_with_kansei: pd.DataFrame,
    ner_metrics_history: List[Dict],
    kansei_insights: Dict,
    lda_results_path: str, # Path to where LDA model/dict/corpus are saved
    output_dir: str
    ):
    logger.info("Creating combined final report...")
    report_path = os.path.join(output_dir, "COMPREHENSIVE_SEAT_ANALYSIS_REPORT.md")
    ensure_directory(output_dir)

    with open(report_path, "w", encoding='utf-8') as f:
        f.write(f"# Comprehensive Seat Analysis Report\n\n")
        f.write(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 1. Overview\n")
        f.write(f"- Total Unique Reviews Analyzed: {processed_df_with_kansei['Feedback Text'].nunique():,}\n")
        f.write(f"- Total Entity Mentions Processed: {len(processed_df_with_kansei):,}\n\n")

        # --- NER Performance Section ---
        f.write(f"## 2. Named Entity Recognition (NER) Performance\n")
        if ner_metrics_history:
            final_ner_metrics = None
            for m in reversed(ner_metrics_history): # Get the last one
                if 'entity_level' in m:
                    final_ner_metrics = m
                    break

            if final_ner_metrics and 'entity_level' in final_ner_metrics:
                el = final_ner_metrics['entity_level']
                f.write(f"### Overall Entity-Level (Final Evaluation):\n")
                f.write(f"- **Precision:** {el.get('precision', 0):.3f}\n")
                f.write(f"- **Recall:** {el.get('recall', 0):.3f}\n")
                f.write(f"- **F1-Score:** {el.get('f1', 0):.3f}\n")
                f.write(f"- True Positives: {el.get('true_positives',0)}, False Positives: {el.get('false_positives',0)}, False Negatives: {el.get('false_negatives',0)}\n\n")

                if 'per_entity' in final_ner_metrics and final_ner_metrics['per_entity']:
                    f.write(f"### Per-Entity Performance (Final Evaluation):\n")
                    f.write("| Entity           | Precision | Recall | F1-Score | Support | TP | FP | FN |\n")
                    f.write("|------------------|-----------|--------|----------|---------|----|----|----|\n")
                    for entity, metrics in sorted(final_ner_metrics['per_entity'].items()):
                        f.write(f"| {entity:<16} | {metrics.get('precision', 0):.3f}     | {metrics.get('recall', 0):.3f}  | {metrics.get('f1', 0):.3f}    | {metrics.get('support', 0):<7} | {metrics.get('tp',0)} | {metrics.get('fp',0)} | {metrics.get('fn',0)} |\n")
                    f.write("\n")
            else:
                f.write("- NER training metrics not fully available for summary.\n")
            f.write("*Refer to `plots/ner_training_performance.png`, `plots/ner_per_label_performance.png`, and `tables/ner_performance_metrics.csv` for more details.*\n\n")
        else:
            f.write("- Custom NER model was not trained or metrics not available.\n\n")

        # --- Transformer Sentiment Analysis Section ---
        f.write(f"## 3. Transformer-Based Sentiment Analysis\n")
        if 'Sentence Sentiment Label' in processed_df_with_kansei.columns:
            overall_sent_dist = processed_df_with_kansei['Sentence Sentiment Label'].value_counts(normalize=True) * 100
            f.write("### Overall Sentence Sentiment Distribution:\n")
            for label, perc in overall_sent_dist.items():
                f.write(f"- **{label.title()}**: {perc:.1f}%\n")
            avg_confidence = processed_df_with_kansei['Sentence Sentiment Score'].mean()
            f.write(f"- Average Sentiment Confidence Score: {avg_confidence:.3f}\n")
            f.write("*Refer to `plots/component_transformer_sentiment.png` for component-specific sentiment.*\n\n")
        else:
            f.write("- Transformer sentiment analysis results not available.\n\n")

        # --- LDA and Kansei Engineering Section ---
        f.write(f"## 4. LDA Topic Modeling and Kansei Engineering Analysis\n")
        f.write(f"LDA was performed to identify underlying topics, which were then mapped to Kansei emotions.\n")
        # Add LDA Coherence if available (requires loading lda_results.json or passing it)
        try:
            with open(os.path.join(lda_results_path, "lda_results.json"), 'r') as lda_f: # Assuming lda_results.json is saved by generate_combined_lda_analysis
                lda_json_results = json.load(lda_f)
                f.write(f"- LDA Coherence Score (c_v): {lda_json_results.get('coherence_score', 'N/A'):.4f}\n")
                f.write(f"- Number of LDA Topics: {lda_json_results.get('num_topics', 'N/A')}\n")
        except FileNotFoundError:
             f.write(f"- LDA coherence score and topic count not found in lda_results.json.\n")
        except Exception as e:
             f.write(f"- Error reading LDA results for report: {e}\n")


        f.write("*Refer to `lda_analysis/lda_visualization.html` and `lda_analysis/lda_topic_terms.csv` for LDA details.*\n\n")

        f.write("### Kansei Emotion Insights:\n")
        if kansei_insights.get('overall_kansei_sentiment'):
            f.write("#### Primary Kansei Emotion Distribution:\n")
            for emotion, data in sorted(kansei_insights['overall_kansei_sentiment'].items(), key=lambda item: item[1]['count'], reverse=True):
                f.write(f"- **{emotion}**: {data['count']} reviews ({data['percentage']:.1f}%), Avg. Confidence: {data['avg_confidence']:.2f}\n")
            f.write("*Refer to `plots/kansei_emotion_distribution.png`.*\n\n")

            f.write("#### Top Kansei Design Recommendations:\n")
            if kansei_insights.get('design_recommendations'):
                for i, rec in enumerate(kansei_insights['design_recommendations'][:5], 1): # Top 5
                    f.write(f"{i}. **Component:** {rec['component']} (Priority: {rec['priority']})\n")
                    f.write(f"   - **Issue:** {rec['issue']}\n")
                    f.write(f"   - **Suggestions:** {'; '.join(rec['suggestions'][:2])}...\n") # First 2 suggestions
            else:
                f.write("- No specific design recommendations generated.\n")
            f.write("\n*Refer to `kansei_design_insights.json` for full details and `plots/feature_kansei_correlation_heatmap.png`.*\n\n")
        else:
            f.write("- Kansei emotion analysis results not available.\n\n")

        f.write(f"## 5. Output Files and Further Exploration\n")
        f.write("Key output files are located in the `combined_nlp_output` directory, under subfolders like `tables`, `plots`, `models`, `lda_analysis`, and `wordclouds`.\n")
        f.write("- **Detailed Data:** `processed_seat_feedback_with_kansei.csv` contains all NER, sentiment, and Kansei results per entity mention.\n")
        f.write("- **NER Model:** Trained NER model is in `models/ner_model`.\n")
        f.write("- **Kansei Insights:** `kansei_design_insights.json`.\n")
        f.write("- **Visualizations:** Various plots provide visual summaries of the findings.\n\n")

    logger.info(f"Combined final report saved to {report_path}")


# --- Main Execution Logic ---
def main_combined_analysis(
    csv_path: str = "final_dataset_compartment.csv",
    text_col: str = "ReviewText", # Original text column name in CSV
    annotations_path: str = "seat_entities_new_min.json",
    output_base_dir: str = "combined_nlp_output",
    train_ner: bool = False, # Changed to False to use pre-trained ultimate model
    ner_iterations: int = 30 # Iterations for NER training
    ):

    logger.info("Starting Combined Seat NLP Analysis Pipeline...")
    ensure_directory(output_base_dir)

    # --- Setup Output Subdirectories ---
    output_models_dir = os.path.join(output_base_dir, "models")
    output_tables_dir = os.path.join(output_base_dir, "tables")
    output_plots_dir = os.path.join(output_base_dir, "plots")
    output_lda_dir = os.path.join(output_base_dir, "lda_analysis")
    output_wordclouds_dir = os.path.join(output_base_dir, "wordclouds")
    ensure_directory(output_models_dir)
    ensure_directory(output_tables_dir)
    ensure_directory(output_plots_dir)
    ensure_directory(output_lda_dir)
    ensure_directory(output_wordclouds_dir)

    # --- Initialize Model Manager ---
    # Path for the NER model to be saved/loaded
    custom_ner_model_path = os.path.join(output_models_dir, "ner_model")
    model_manager = ModelManager(custom_ner_model_path=custom_ner_model_path)

    # --- Data Loading and Preprocessing ---
    try:
        full_df, test_data_texts = process_data_efficiently(csv_path, text_col)
        # 'full_df' now contains the original text column and 'clean_text'
        if test_data_texts:
            pd.DataFrame({"test_texts": test_data_texts}).to_csv(os.path.join(output_tables_dir, "test_data_samples.csv"), index=False)
    except Exception as e:
        logger.error(f"Critical error during data loading: {e}", exc_info=True)
        return

    # --- NER Training (if enabled) ---
    ner_trainer = None
    ner_metrics_history = []
    trained_ner_model_spacy = None

    if train_ner:
        logger.info("NER training is enabled.")
        training_data = load_training_data(annotations_path)
        if training_data and len(training_data) >= 5: # Min examples for training
            ner_trainer = NERTrainer(training_data)
            trained_ner_model_spacy = ner_trainer.train_model(n_iter=ner_iterations)
            if trained_ner_model_spacy:
                logger.info("NER model training completed.")
                ner_trainer.save_model_and_metrics(
                    model_path=custom_ner_model_path,
                    metrics_path=os.path.join(output_models_dir, "ner_training_metrics.json")
                )
                ner_metrics_history = ner_trainer.metrics_history
                generate_ner_metrics_visualization(ner_metrics_history, output_base_dir) # Pass base_dir for plots subdir
                # Save NER performance metrics as CSV
                if ner_metrics_history:
                    final_metrics = ner_metrics_history[-1] # Get the last (final) evaluation metrics
                    ner_perf_data = []
                    if 'entity_level' in final_metrics:
                         el = final_metrics['entity_level']
                         ner_perf_data.append({'Metric_Type': 'Overall_Entity', 'Label': 'ALL', **el})
                    if 'per_entity' in final_metrics:
                        for label, metrics in final_metrics['per_entity'].items():
                            ner_perf_data.append({'Metric_Type': 'Per_Entity', 'Label': label, **metrics})
                    pd.DataFrame(ner_perf_data).to_csv(os.path.join(output_tables_dir, "ner_performance_metrics.csv"), index=False)

            else:
                logger.warning("NER model training failed or produced no model. Using default NER from ModelManager.")
                trained_ner_model_spacy = model_manager.nlp_model # Fallback to default (blank or preloaded)
        else:
            logger.warning(f"Insufficient training data ({len(training_data)} examples found at '{annotations_path}'). Skipping NER training. Using default NER.")
            trained_ner_model_spacy = model_manager.nlp_model # Fallback
    else:
        logger.info("NER training is disabled. Loading NER model (if exists) or using blank model via ModelManager.")
        trained_ner_model_spacy = model_manager.nlp_model # Load existing or blank

    if not trained_ner_model_spacy: # Ensure we have some spacy model object
        logger.error("No spaCy NER model available (neither trained nor loaded). Cannot proceed with NER.")
        return

    # --- Batch Processing for NER and Sentiment ---
    logger.info("Processing texts for NER and Sentiment...")
    batch_processor = BatchNLPProcessor(
        nlp_model=trained_ner_model_spacy, # Use the trained or loaded NER model
        sentiment_model=model_manager.sentiment_model
    )
    # Pass both cleaned text for processing and original text for reporting
    processed_ner_sentiment_results = batch_processor.process_texts_batch(
        full_df['clean_text'].tolist(),
        full_df[text_col].tolist() # Original text column
    )

    if not processed_ner_sentiment_results:
        logger.warning("No results from NER/Sentiment batch processing. Further analysis might be limited.")
        processed_ner_sentiment_df = pd.DataFrame()
    else:
        processed_ner_sentiment_df = pd.DataFrame(processed_ner_sentiment_results)
        processed_ner_sentiment_df.to_csv(os.path.join(output_tables_dir, "intermediate_ner_sentiment_results.csv"), index=False)

    generate_general_visualizations(processed_ner_sentiment_df, output_base_dir)


    # --- LDA Topic Modeling ---
    # Use 'clean_text' for LDA as it's more uniform
    lda_model, lda_dictionary, lda_corpus = generate_combined_lda_analysis(
        full_df=full_df,
        text_column_for_lda='clean_text',
        nlp_core_model=model_manager.nlp_core_for_lda, # Use separate model for LDA preprocessing
        num_topics=10, # Or make this configurable
        output_dir=output_lda_dir
    )

    # --- Kansei Engineering Analysis ---
    kansei_insights_data = {}
    df_with_kansei = processed_ner_sentiment_df.copy() # Start with NER/Sentiment results

    if lda_model and lda_dictionary and lda_corpus and not processed_ner_sentiment_df.empty:
        logger.info("Proceeding with Kansei Engineering Analysis...")
        kansei_module = KanseiModule(
            lda_model=lda_model,
            dictionary=lda_dictionary,
            corpus=lda_corpus, # This corpus is from generate_combined_lda_analysis
            full_df_with_ner_sentiment=processed_ner_sentiment_df # Pass df with NER/Sentiment
        )

        # map_topics_to_kansei returns one emotion set per original review text
        kansei_results_per_review = kansei_module.map_topics_to_kansei()

        if kansei_results_per_review:
            # Generate insights and merge Kansei emotions back to the main DataFrame
            kansei_insights_data, df_with_kansei = kansei_module.generate_design_insights(
                kansei_results_per_review,
                processed_ner_sentiment_df # Original df with NER/Sentiment
            )

            # Save Kansei insights to JSON
            with open(os.path.join(output_base_dir, "kansei_design_insights.json"), 'w', encoding='utf-8') as f:
                # Helper to convert complex objects like Counter to dict for JSON
                def make_serializable(obj):
                    if isinstance(obj, (Counter, defaultdict)):
                        return dict(obj)
                    if isinstance(obj, dict):
                        return {k: make_serializable(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [make_serializable(i) for i in obj]
                    if isinstance(obj, np.integer): return int(obj)
                    if isinstance(obj, np.floating): return float(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    return obj
                json.dump(make_serializable(kansei_insights_data), f, indent=2)

            generate_kansei_visualizations(kansei_insights_data, output_base_dir)
            logger.info("Kansei analysis complete. Insights and visualizations saved.")
        else:
            logger.warning("Kansei mapping did not produce results. Skipping further Kansei analysis.")
    else:
        logger.warning("LDA model/dictionary/corpus not available or no NER/Sentiment data. Skipping Kansei Engineering Analysis.")

    # --- Save Final Combined DataFrame ---
    # df_with_kansei should now have NER, Sentiment, and Kansei info (if Kansei ran)
    final_output_csv_path = os.path.join(output_base_dir, "processed_seat_feedback_with_kansei.csv")
    df_with_kansei.to_csv(final_output_csv_path, index=False)
    logger.info(f"Final combined data saved to: {final_output_csv_path}")

    # --- Generate Final Report ---
    create_combined_report(
        processed_df_with_kansei=df_with_kansei,
        ner_metrics_history=ner_metrics_history,
        kansei_insights=kansei_insights_data,
        lda_results_path=output_lda_dir, # Path to where lda_results.json is
        output_dir=output_base_dir
    )

    gc.collect()
    logger.info("="*50)
    logger.info("Combined NLP Analysis Pipeline Completed Successfully!")
    logger.info(f"All outputs saved in: {output_base_dir}")
    logger.info("="*50)

if __name__ == "__main__":
    # Configuration (can be externalized to a config file or CLI args)
    DATASET_CSV_PATH = 'final_dataset_compartment.csv'
    TEXT_COLUMN_NAME = 'ReviewText' # The column in your CSV with the review texts
    ANNOTATIONS_JSON_PATH = 'seat_entities_new_min.json' # Your NER annotations
    OUTPUT_DIRECTORY = 'combined_nlp_output_v2' # Changed to avoid overwriting previous runs

    # Set to True to retrain the NER model, False to try loading an existing one or use a blank model
    SHOULD_TRAIN_NER = True
    NER_TRAINING_ITERATIONS = 35 # Reduced for quicker test runs, increase for better models (e.g., 30-50)

    # Ensure NLTK components are downloaded (moved here for clarity)
    try:
        nltk.data.find('corpora/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('taggers/averaged_perceptron_tagger') # For POS tagging in LDA if en_core_web_sm not used
        nltk.data.find('corpora/wordnet') # For lemmatization in LDA if en_core_web_sm not used
    except LookupError as e:
        logger.info(f"Downloading missing NLTK resource: {e.args[0]}")
        nltk.download(e.args[0].split('/')[-1] if '/' in e.args[0] else e.args[0], quiet=True)


    main_combined_analysis(
        csv_path=DATASET_CSV_PATH,
        text_col=TEXT_COLUMN_NAME,
        annotations_path=ANNOTATIONS_JSON_PATH,
        output_base_dir=OUTPUT_DIRECTORY,
        train_ner=SHOULD_TRAIN_NER,
        ner_iterations=NER_TRAINING_ITERATIONS
    )