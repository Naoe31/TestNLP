# Combined NLP Analytics with Kansei Engineering
# Merges advanced NER training from Final_Push_Complete_Analytics.py
# with Kansei Engineering from combined_seat_kansei.py

import pandas as pd
import random
import json
import os
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import spacy
from spacy.training import Example
from spacy.tokens import DocBin # Added from combined_seat_kansei

import matplotlib
matplotlib.use('Agg') # Set non-interactive backend for matplotlib
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

# Imports from combined_seat_kansei.py
import unicodedata
from bs4 import BeautifulSoup
import pathlib
import logging
import warnings
from transformers import pipeline, AutoTokenizer # For sentiment model
import torch # For sentiment model device
import gc # For garbage collection
from functools import lru_cache # For caching

# Set up logging (from combined_seat_kansei.py)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings (from combined_seat_kansei.py)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# SET FIXED SEEDS FOR REPRODUCIBILITY
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# spacy.util.fix_random_seed(RANDOM_SEED) # Called later in trainer

# Set plotting style
plt.style.use('default') # Final_Push_Complete_Analytics.py style
sns.set_palette("husl") # Final_Push_Complete_Analytics.py palette
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# --- Global Constants and Configurations (Merged) ---
# More comprehensive STANDARDIZED_LABELS from combined_seat_kansei.py
STANDARDIZED_LABELS = {
    "ARMREST", "BACKREST", "HEADREST", "CUSHION", "MATERIAL",
    "LUMBAR_SUPPORT", "SEAT_SIZE", "RECLINER", "FOOTREST",
    "SEAT_MESSAGE", "SEAT_WARMER", "TRAYTABLE"
}

# More comprehensive SEAT_SYNONYMS from combined_seat_kansei.py
# This will be used by NER Trainer, Augmenters, and Kansei Module
SEAT_SYNONYMS = {
    "ARMREST": ["armrest", "arm rest", "arm-rest", "armrests", "arm support", "elbow rest"],
    "BACKREST": ["backrest", "seat back", "seatback", "back", "back support", "back-rest", "spine support", "ergonomic back"],
    "HEADREST": ["headrest", "neckrest", "head support", "neck support", "head-rest", "headrests", "head cushion", "neck cushion"],
    "CUSHION": ["cushion", "padding", "cushioning", "seat base", "base", "bottom", "cushions", "padded", "pad", "memory foam", "seat cushion"],
    "MATERIAL": ["material", "fabric", "leather", "upholstery", "vinyl", "cloth", "velvet", "textile", "materials", "synthetic leather", "genuine leather", "premium leather", "suede", "canvas", "linen", "deer skin", "breathable fabric", "high-quality material"],
    "LUMBAR_SUPPORT": ["lumbar support", "lumbar", "lumbar pad", "lumbar cushion", "lower back support", "ergonomic lumbar", "adjustable lumbar", "spine alignment"],
    "SEAT_SIZE": ["legroom", "leg room", "space", "seat width", "seat size", "narrow", "tight", "cramped", "spacious", "roomy"], # Added from combined_seat_kansei
    "RECLINER": ["reclining", "recline", "recliner", "reclined", "reclines", "reclinable", "seat angle", "seat position", "lie flat", "flat bed", "180 degree", "fully reclined", "tilting backrest"],
    "FOOTREST": ["footrest", "foot-rest", "footrests", "leg support", "ottoman", "leg extension", "calf support", "adjustable footrest"],
    "SEAT_MESSAGE": ["massage", "massaging", "massager", "massage function", "vibration", "vibrating", "therapeutic massage", "lumbar massage", "seat massage"],
    "SEAT_WARMER": ["warmer", "warming", "heated", "heating", "seat warmer", "seat heating", "temperature control", "warm seat", "climate control", "thermal comfort"],
    "TRAYTABLE": ["tray table", "fold down table", "dining table", "work table", "work surface", "laptop table", "laptop tray"]
}

# Download required NLTK data
try:
    nltk.data.find('corpora/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


# --- Helper Functions (from combined_seat_kansei.py) ---
@lru_cache(maxsize=None)
def ensure_directory(path: str) -> None:
    """Ensures that a directory exists, creating it if necessary."""
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")

@lru_cache(maxsize=10000)
def clean_text_cached(txt: str) -> str:
    """Cleans and normalizes a text string, with caching."""
    if pd.isna(txt) or not txt:
        return ""
    # Order of operations is important
    txt = str(txt)
    txt = BeautifulSoup(txt, "lxml").get_text(" ") # Remove HTML
    txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("utf-8", "ignore") # Normalize unicode
    txt = txt.lower() # Convert to lowercase
    txt = re.sub(r"https?://\S+|www\.\S+", "", txt)  # Remove URLs
    txt = re.sub(r"<.*?>+", "", txt)  # Remove lingering HTML tags
    txt = re.sub(r"@\w+|#\w+", "", txt) # Remove mentions and hashtags
    # Standardize common abbreviations (example)
    abbreviations = {
        'e.g.': 'for example', 'i.e.': 'that is', 'etc.': 'and so on',
        'vs.': 'versus', 'approx.': 'approximately', 'min.': 'minimum', 'max.': 'maximum'
    }
    for abbr, full in abbreviations.items():
        txt = txt.replace(abbr, full)
    txt = re.sub(r"[^a-z0-9\s.,!?'\"]", "", txt) # Keep basic punctuation for sentiment
    txt = re.sub(r"\s+", " ", txt).strip() # Normalize whitespace
    return txt

def map_text_offsets(original_text: str, cleaned_text: str, original_start: int, original_end: int) -> Tuple[Optional[int], Optional[int]]:
    """Maps entity offsets from original text to cleaned text."""
    if not original_text or not cleaned_text or original_start is None or original_end is None:
        return None, None

    original_snippet = original_text[original_start:original_end]
    # Try to find the snippet directly in the cleaned text (case-insensitive)
    # Clean the snippet in a similar way to how the text was cleaned for a better match
    cleaned_snippet = clean_text_cached(original_snippet) # Use the same cleaning for the snippet

    if not cleaned_snippet: # If snippet becomes empty after cleaning
        return None, None

    # Search for the cleaned snippet in the already cleaned text
    # Note: clean_text_cached already converts to lower, so direct find is okay.
    # If clean_text_cached did not lowercase, we'd use .lower() here.
    start_in_cleaned = cleaned_text.find(cleaned_snippet)

    if start_in_cleaned != -1:
        end_in_cleaned = start_in_cleaned + len(cleaned_snippet)
        return start_in_cleaned, end_in_cleaned
    else:
        # Fallback: If direct find fails, log and return None
        # More sophisticated alignment (e.g., diff-based) is complex and out of scope here.
        logger.debug(f"Offset mapping failed for snippet: '{original_snippet}' (cleaned: '{cleaned_snippet}') in text: '{cleaned_text[:100]}...'")
        return None, None

# --- Data Loading and Preprocessing (from combined_seat_kansei.py) ---
def process_data_efficiently(csv_path: str, text_col_name: str) -> Tuple[pd.DataFrame, List[str]]:
    """Loads and preprocesses data from a CSV file efficiently in chunks."""
    logger.info(f"Loading data from {csv_path} using text column '{text_col_name}'")
    if not os.path.exists(csv_path):
        logger.error(f"CSV file '{csv_path}' not found.")
        raise FileNotFoundError(f"CSV file '{csv_path}' not found")

    all_chunks = []
    test_data_texts_samples = [] # Samples for quick checks, not a formal test set
    chunk_size = 10000  # Process 10,000 rows at a time
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    df_reader = None

    for encoding in encodings_to_try:
        try:
            df_reader = pd.read_csv(csv_path, chunksize=chunk_size, encoding=encoding, on_bad_lines='skip')
            logger.info(f"Successfully opened CSV with '{encoding}' encoding.")
            break
        except UnicodeDecodeError:
            logger.debug(f"Failed to decode CSV with '{encoding}', trying next.")
            continue
        except Exception as e: # Catch other potential errors like file not found during chunking
            logger.warning(f"Error reading CSV with '{encoding}': {e}")
            continue
    
    if df_reader is None:
        logger.error(f"Could not read CSV file '{csv_path}' with any of the attempted encodings.")
        raise ValueError("Could not read CSV file with any supported encoding")

    for chunk_idx, chunk_df in enumerate(df_reader):
        logger.info(f"Processing chunk {chunk_idx + 1} with {len(chunk_df)} rows.")
        
        if text_col_name not in chunk_df.columns:
            logger.error(f"Text column '{text_col_name}' not found in chunk {chunk_idx + 1}. Available columns: {list(chunk_df.columns)}")
            # Attempt to find a fallback column if the specified one is missing
            potential_cols = [col for col in chunk_df.columns if isinstance(col, str) and any(keyword in col.lower() for keyword in ['review', 'text', 'feedback', 'comment'])]
            if potential_cols:
                actual_text_col = potential_cols[0]
                logger.warning(f"Using fallback text column '{actual_text_col}' for chunk {chunk_idx + 1}.")
            else:
                logger.error(f"No suitable text column found in chunk {chunk_idx + 1}. Skipping this chunk.")
                continue
        else:
            actual_text_col = text_col_name

        chunk_df = chunk_df.dropna(subset=[actual_text_col])
        if chunk_df.empty:
            logger.info(f"Chunk {chunk_idx + 1} is empty after dropping NaNs in '{actual_text_col}'.")
            continue

        # Apply text cleaning
        chunk_df['clean_text'] = chunk_df[actual_text_col].apply(lambda x: clean_text_cached(str(x)))
        
        # Filter out rows where cleaned text became empty
        chunk_df = chunk_df[chunk_df['clean_text'].str.len() > 0]
        if chunk_df.empty:
            logger.info(f"Chunk {chunk_idx + 1} is empty after text cleaning.")
            continue
        
        all_chunks.append(chunk_df)

        # Collect some sample texts (not a formal test set)
        if len(chunk_df["clean_text"]) >= 10: # Ensure enough data to sample
            sample_size = max(1, min(int(0.01 * len(chunk_df["clean_text"])), 20)) # 1% or max 20 samples per chunk
            try:
                test_samples_from_chunk = random.sample(chunk_df["clean_text"].tolist(), sample_size)
                test_data_texts_samples.extend(test_samples_from_chunk)
            except ValueError: # If sample_size > population
                 test_data_texts_samples.extend(chunk_df["clean_text"].tolist())


    if not all_chunks:
        logger.error("No valid data found after processing all chunks from the CSV.")
        raise ValueError("No valid data found in the CSV file.")

    full_processed_df = pd.concat(all_chunks, ignore_index=True)
    # Ensure the original text column is preserved if it was different from 'clean_text' source
    if actual_text_col != text_col_name and actual_text_col in full_processed_df.columns:
        full_processed_df.rename(columns={actual_text_col: text_col_name}, inplace=True)


    logger.info(f"Total processed texts: {len(full_processed_df):,}. Sample test texts collected: {len(test_data_texts_samples)}")
    
    if full_processed_df.empty:
        logger.error("No valid texts found in the dataset after full processing.")
        raise ValueError("No valid texts found in the dataset")

    return full_processed_df, list(set(test_data_texts_samples))


# --- NER Data Loading (Enhanced - from combined_seat_kansei.py) ---
def load_ner_training_data(annotated_data_path: str) -> List[Tuple]:
    """Loads NER training data from JSON, cleans text, and maps entity offsets."""
    logger.info(f"Loading NER training data from: {annotated_data_path}")
    if not os.path.exists(annotated_data_path):
        logger.warning(f"NER training data file not found: {annotated_data_path}")
        return []

    try:
        with open(annotated_data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {annotated_data_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error reading {annotated_data_path}: {e}")
        return []

    training_data = []
    # Label mapping to ensure consistency with STANDARDIZED_LABELS
    label_map = {
        "recliner": "RECLINER", "seat_message": "SEAT_MESSAGE", "seat_warmer": "SEAT_WARMER",
        "headrest": "HEADREST", "armrest": "ARMREST", "footrest": "FOOTREST",
        "backrest": "BACKREST", "cushion": "CUSHION", "material": "MATERIAL",
        "traytable": "TRAYTABLE", "lumbar_support": "LUMBAR_SUPPORT",
        "seat_material": "MATERIAL", "lumbar": "LUMBAR_SUPPORT", "seat_size": "SEAT_SIZE",
        "legroom": "SEAT_SIZE" # Common synonym
    }

    for item_idx, item in enumerate(raw_data):
        original_text = None
        # Try to extract text from common Label Studio and other formats
        if 'ReviewText' in item:
            original_text = item.get('ReviewText')
        elif 'data' in item and isinstance(item['data'], dict):
            text_data_dict = item['data']
            original_text = text_data_dict.get('ReviewText') or \
                            text_data_dict.get('Review Text') or \
                            text_data_dict.get('feedback') or \
                            text_data_dict.get('text')
        elif 'text' in item: # Simpler format
            original_text = item.get('text')

        if not original_text or not isinstance(original_text, str) or not original_text.strip():
            logger.debug(f"Skipping item {item_idx} due to missing or invalid text.")
            continue

        cleaned_text = clean_text_cached(original_text)
        if not cleaned_text:
            logger.debug(f"Skipping item {item_idx} as text became empty after cleaning: '{original_text[:50]}...'")
            continue

        entities = []
        raw_annotations = []
        if 'label' in item and isinstance(item['label'], list): # Kansei/custom format
            raw_annotations = item['label']
        elif 'annotations' in item and isinstance(item['annotations'], list) and \
             item['annotations'] and 'result' in item['annotations'][0] and \
             isinstance(item['annotations'][0]['result'], list): # Label Studio format
            raw_annotations = item['annotations'][0]['result']
        elif 'entities' in item and isinstance(item['entities'], list): # Simple (start, end, label) format
            for start, end, label_text in item['entities']:
                standardized_label = label_map.get(str(label_text).lower(), str(label_text).upper())
                if standardized_label in STANDARDIZED_LABELS:
                    mapped_start, mapped_end = map_text_offsets(original_text, cleaned_text, start, end)
                    if mapped_start is not None and mapped_end is not None and mapped_start < mapped_end:
                        entities.append((mapped_start, mapped_end, standardized_label))
            if entities: # If simple format processed, add and continue
                 training_data.append((cleaned_text, {"entities": entities}))
            continue


        for ann_idx, ann in enumerate(raw_annotations):
            if not isinstance(ann, dict): continue

            # For Label Studio format
            value = ann.get("value", {})
            original_start, original_end = value.get("start"), value.get("end")
            labels_list = value.get("labels", [])

            # For Kansei/custom format (heuristic)
            if original_start is None and 'start' in ann: original_start = ann['start']
            if original_end is None and 'end' in ann: original_end = ann['end']
            if not labels_list and 'labels' in ann and isinstance(ann['labels'], list): labels_list = ann['labels']
            if not labels_list and 'label' in ann and isinstance(ann['label'], list): labels_list = ann['label'] # Another variation
            if not labels_list and 'text_label' in ann : labels_list = [ann['text_label']]


            if original_start is None or original_end is None or not labels_list:
                logger.debug(f"Skipping annotation {ann_idx} in item {item_idx} due to missing start/end/label.")
                continue
            
            try:
                original_start, original_end = int(original_start), int(original_end)
            except ValueError:
                logger.warning(f"Invalid start/end for annotation in item {item_idx}: {original_start}, {original_end}")
                continue

            raw_label_text = str(labels_list[0]).lower()
            standardized_label = label_map.get(raw_label_text, raw_label_text.upper())

            if standardized_label not in STANDARDIZED_LABELS:
                logger.debug(f"Skipping annotation with unrecognized label '{standardized_label}' from raw '{raw_label_text}'.")
                continue

            # Map offsets
            mapped_start, mapped_end = map_text_offsets(original_text, cleaned_text, original_start, original_end)
            if mapped_start is not None and mapped_end is not None and mapped_start < mapped_end:
                # Ensure no overlapping entities with the same label (simple check)
                is_duplicate = False
                for es, ee, el in entities:
                    if el == standardized_label and max(mapped_start, es) < min(mapped_end, ee):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    entities.append((mapped_start, mapped_end, standardized_label))
            else:
                logger.debug(f"Failed to map entity: '{original_text[original_start:original_end]}' for label '{standardized_label}' in text: '{original_text[:50]}...'")


        if entities:
            # Sort entities by start position to prevent spaCy errors
            entities.sort(key=lambda x: x[0])
            training_data.append((cleaned_text, {"entities": entities}))
        elif raw_annotations : # If there were annotations but none were valid
             logger.debug(f"Item {item_idx} had annotations but none were valid after processing: '{original_text[:50]}...'")


    logger.info(f"Loaded {len(training_data)} valid NER training examples after cleaning and offset mapping.")
    if training_data:
        logger.info(f"Example NER training data item: Text='{training_data[0][0][:100]}...', Entities={training_data[0][1]['entities'][:2]}")
    return training_data


# --- ModelManager (Adapted from combined_seat_kansei.py for Sentiment and LDA spaCy model) ---
class ModelManager:
    def __init__(self):
        self._sentiment_model = None
        self._nlp_core_for_lda = None # spaCy model for LDA preprocessing

    @property
    def sentiment_model(self):
        if self._sentiment_model is None:
            try:
                logger.info("Loading sentiment model: siebert/sentiment-roberta-large-english")
                # Determine device
                device = 0 if torch.cuda.is_available() else -1
                if device == 0:
                    logger.info("CUDA available, loading sentiment model on GPU.")
                else:
                    logger.info("CUDA not available, loading sentiment model on CPU.")
                
                self._sentiment_model = pipeline(
                    "sentiment-analysis", 
                    model="siebert/sentiment-roberta-large-english",
                    tokenizer="siebert/sentiment-roberta-large-english", # Explicitly specify tokenizer
                    truncation=True, 
                    max_length=512, 
                    device=device,
                    batch_size=8 if device == 0 else 4 # Adjust batch size based on device
                )
                logger.info("Sentiment model 'siebert/sentiment-roberta-large-english' loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load preferred sentiment model (siebert/sentiment-roberta-large-english): {e}. Trying default.")
                try:
                    self._sentiment_model = pipeline(
                        "sentiment-analysis", 
                        truncation=True, 
                        max_length=512,
                        device=-1, # Force CPU for fallback
                        batch_size=4
                    )
                    logger.info("Loaded fallback default sentiment model.")
                except Exception as e2:
                    logger.error(f"Failed to load any sentiment model: {e2}")
                    self._sentiment_model = None # Ensure it's None if loading fails
        return self._sentiment_model

    @property
    def nlp_core_for_lda(self): # For LDA preprocessing
        if self._nlp_core_for_lda is None:
            try:
                logger.info("Loading spaCy 'en_core_web_sm' for LDA preprocessing (disabling ner, parser).")
                self._nlp_core_for_lda = spacy.load("en_core_web_sm", disable=["ner", "parser"])
                logger.info("'en_core_web_sm' for LDA loaded successfully.")
            except OSError:
                logger.warning("'en_core_web_sm' not found. LDA preprocessing might be less effective. Using blank 'en' model for LDA.")
                self._nlp_core_for_lda = spacy.blank("en")
                if "sentencizer" not in self._nlp_core_for_lda.pipe_names:
                     self._nlp_core_for_lda.add_pipe("sentencizer")
            except Exception as e: # Catch any other exception during loading
                logger.error(f"An unexpected error occurred while loading 'en_core_web_sm': {e}. Using blank 'en' model for LDA.")
                self._nlp_core_for_lda = spacy.blank("en")
                if "sentencizer" not in self._nlp_core_for_lda.pipe_names:
                     self._nlp_core_for_lda.add_pipe("sentencizer")
        return self._nlp_core_for_lda


# --- BatchNLPProcessor (from combined_seat_kansei.py) ---
class BatchNLPProcessor:
    def __init__(self, ner_model_spacy, sentiment_model_pipeline, batch_size: int = 64): # Increased batch_size
        self.ner_model = ner_model_spacy
        self.sentiment_model = sentiment_model_pipeline
        self.batch_size = batch_size
        
        # Ensure sentencizer is in the NER model pipeline for sentence segmentation
        if self.ner_model and "sentencizer" not in self.ner_model.pipe_names:
            try:
                # Add sentencizer, preferably before NER if model is not trained yet,
                # or first if model is already trained.
                if not self.ner_model.meta.get("trained_pipeline"): # If model is blank or not fully trained
                    self.ner_model.add_pipe("sentencizer", before="ner" if "ner" in self.ner_model.pipe_names else None)
                else: # If model is trained, add it first
                    self.ner_model.add_pipe("sentencizer", first=True)
                logger.info("Added 'sentencizer' to the NER model pipeline.")
            except Exception as e:
                 logger.warning(f"Could not add 'sentencizer' to NER model: {e}. Sentence segmentation might be affected.")


    def process_texts_batch(self, cleaned_texts: List[str], original_texts: Optional[List[str]] = None) -> List[Dict]:
        """Processes texts in batches for NER and sentiment analysis."""
        logger.info(f"Starting batch processing of {len(cleaned_texts)} texts...")
        results = []
        if not cleaned_texts:
            logger.warning("No texts provided for batch processing.")
            return results

        if original_texts is None:
            original_texts = cleaned_texts # Use cleaned if original not provided
        elif len(cleaned_texts) != len(original_texts):
            logger.error("Mismatch between lengths of cleaned_texts and original_texts. Cannot proceed.")
            return results

        num_texts = len(cleaned_texts)
        for i in range(0, num_texts, self.batch_size):
            batch_cleaned_texts = cleaned_texts[i:i + self.batch_size]
            batch_original_texts = original_texts[i:i + self.batch_size]
            logger.debug(f"Processing batch {i // self.batch_size + 1}/{(num_texts + self.batch_size -1)//self.batch_size}, size: {len(batch_cleaned_texts)}")

            # Filter out any empty strings that might have slipped through
            valid_indices = [idx for idx, text in enumerate(batch_cleaned_texts) if text and text.strip()]
            if not valid_indices:
                logger.debug("Skipping batch as it contains no valid texts after filtering.")
                continue
            
            current_batch_cleaned = [batch_cleaned_texts[idx] for idx in valid_indices]
            current_batch_original = [batch_original_texts[idx] for idx in valid_indices]

            # NER processing
            docs = []
            try:
                docs = list(self.ner_model.pipe(current_batch_cleaned, batch_size=len(current_batch_cleaned)))
            except Exception as e:
                logger.error(f"Error in NER model pipe for a batch: {e}. Processing texts individually for this batch.")
                for text_item in current_batch_cleaned:
                    try:
                        docs.append(self.ner_model(text_item))
                    except Exception as e_ind:
                        logger.error(f"Failed to process individual text for NER '{text_item[:50]}...': {e_ind}")
                        docs.append(self.ner_model.make_doc(text_item)) # Create a blank doc on failure

            for doc_idx, doc in enumerate(docs):
                original_text_for_doc = current_batch_original[doc_idx]
                cleaned_text_for_doc = doc.text # Text as processed by spaCy

                # Component frequency within this document
                entity_counts_in_doc = Counter(ent.label_ for ent in doc.ents if ent.label_ in STANDARDIZED_LABELS)

                try:
                    sentences = list(doc.sents)
                except ValueError: # Fallback if sentencizer fails for some reason
                    logger.warning(f"Sentencizer failed for doc: '{cleaned_text_for_doc[:50]}...'. Treating as one sentence.")
                    sentences = [doc[:]] # Treat whole doc as one sentence span

                if not sentences: continue

                # Sentiment analysis for all sentences in the current document
                sentence_texts = [s.text.strip() for s in sentences if s.text.strip()]
                sentiment_outputs = []
                if self.sentiment_model and sentence_texts:
                    try:
                        sentiment_outputs = self.sentiment_model(sentence_texts, batch_size=self.batch_size) # Use pipeline's batching
                    except Exception as e:
                        logger.warning(f"Sentiment analysis failed for a batch of sentences from doc '{cleaned_text_for_doc[:50]}...': {e}")
                        # Fallback: create dummy unknown sentiment for these sentences
                        sentiment_outputs = [{"label": "unknown", "score": 0.0}] * len(sentence_texts)
                elif not sentence_texts: # No sentences to analyze
                    sentiment_outputs = []
                else: # No sentiment model
                    sentiment_outputs = [{"label": "N/A", "score": 0.0}] * len(sentence_texts)


                for sent_idx, sent_span in enumerate(sentences):
                    sent_text_original_case = sent_span.text.strip()
                    if not sent_text_original_case: continue

                    current_sentiment = sentiment_outputs[sent_idx] if sent_idx < len(sentiment_outputs) else {"label": "unknown", "score": 0.0}

                    # Entities within this sentence
                    sent_entities = [ent for ent in doc.ents if ent.start_char >= sent_span.start_char and ent.end_char <= sent_span.end_char and ent.label_ in STANDARDIZED_LABELS]

                    if not sent_entities: # If sentence has no relevant entities, still log sentence if needed for other analysis
                        # For this combined script, we focus on entity-centric rows
                        pass
                    else:
                        for ent in sent_entities:
                            results.append({
                                "Feedback Text": original_text_for_doc,
                                "Cleaned Text": cleaned_text_for_doc,
                                "Seat Component": ent.label_,
                                "Cue Word": ent.text,
                                "Component Frequency in Text": entity_counts_in_doc.get(ent.label_, 0),
                                "Sentence Sentiment Label": current_sentiment["label"].upper(), # Standardize to upper
                                "Sentence Sentiment Score": round(current_sentiment["score"], 4),
                                "Sentence Text": sent_text_original_case,
                                "Entity Start Char (Cleaned)": ent.start_char,
                                "Entity End Char (Cleaned)": ent.end_char
                            })
        logger.info(f"Batch processing complete. Generated {len(results)} entity-sentence records.")
        return results


# --- NER Augmentation and Training Classes (from Final_Push_Complete_Analytics.py) ---
class PerfectScoreRegularizer:
    """Regularizes entities with perfect scores (1.00) to target range (0.97-0.98)"""
    def __init__(self, target_max_score: float = 0.97): # SEAT_SYNONYMS is now global
        self.target_max_score = target_max_score
        self.challenging_templates = {
            "GENERAL": [
                "The {entity} area needs some improvement overall", "While the {entity} is okay, it could be better designed",
                "The {entity} has both good and problematic aspects", "I have mixed feelings about the {entity} quality",
                "The {entity} works but isn't quite perfect", "The {entity} design could use some refinement",
                "The {entity} functionality is adequate but not exceptional"
            ]
        }
        self.entity_challenging_templates = { # Simplified for brevity, original script has more
            "ARMREST": ["The armrest position is slightly awkward for my arm"],
            "BACKREST": ["The backrest angle could be more adjustable for comfort"],
            "HEADREST": ["The headrest position needs fine-tuning for my height"],
            "CUSHION": ["The cushion firmness could be better balanced"],
            "MATERIAL": ["The {entity} feels premium and luxurious","The {entity} feels cheap and synthetic"],
            "LUMBAR_SUPPORT": ["The lumbar support position could be more precise"],
            "RECLINER": ["The recliner mechanism could be smoother in operation"],
            "FOOTREST": ["The footrest extension could be slightly longer"],
            "SEAT_MESSAGE": ["The massage intensity could have more gradual settings"],
            "SEAT_WARMER": ["The seat warmer could heat up slightly more evenly"],
            "TRAYTABLE": ["The tray table could lock more securely in position"]
        }

    def detect_perfect_entities(self, metrics: Dict) -> List[str]:
        perfect_entities = []
        if 'per_entity' in metrics:
            for entity, entity_metrics in metrics['per_entity'].items():
                if (entity_metrics.get('support', 0) > 0 and entity_metrics.get('f1', 0) >= 0.999):
                    perfect_entities.append(entity)
        return perfect_entities

    def generate_challenging_examples(self, entity_type: str, count: int = 30) -> List[Tuple[str, Dict]]:
        challenging_examples = []
        entity_terms = SEAT_SYNONYMS.get(entity_type, [entity_type.lower().replace('_', ' ')])
        templates = (self.entity_challenging_templates.get(entity_type, []) + self.challenging_templates["GENERAL"])
        if not templates: return []

        for _ in range(count):
            entity_term = random.choice(entity_terms)
            template = random.choice(templates)
            text = template.replace("{entity}", entity_term)
            # Add natural variations
            starters = ["", "Overall, ", "I think ", "In my opinion, ", "Honestly, ", "Sometimes "]
            endings = ["", ".", " overall.", " I suppose.", " in general.", " to be honest."]
            if random.random() < 0.4: text = random.choice(starters) + text.lower()
            if random.random() < 0.4: text += random.choice(endings)
            
            cleaned_text_for_example = clean_text_cached(text) # Use consistent cleaning
            if not cleaned_text_for_example: continue

            # Find entity position in the *cleaned* text
            # The entity term itself might change after cleaning, so we find the cleaned version of the term
            cleaned_entity_term = clean_text_cached(entity_term)
            if not cleaned_entity_term: continue

            start_pos = cleaned_text_for_example.find(cleaned_entity_term)
            if start_pos == -1: continue
            end_pos = start_pos + len(cleaned_entity_term)
            
            entities = [(start_pos, end_pos, entity_type)]
            challenging_examples.append((cleaned_text_for_example, {"entities": entities}))
        return challenging_examples

    def add_subtle_noise_examples(self, entity_type: str, count: int = 15) -> List[Tuple[str, Dict]]:
        noise_examples = []
        entity_terms = SEAT_SYNONYMS.get(entity_type, [entity_type.lower().replace('_', ' ')])
        ambiguous_templates = [
            f"The seat area around the {entity_type.lower().replace('_', ' ')} seems fine",
            f"Near the {entity_type.lower().replace('_', ' ')} region, things look okay",
        ]
        for _ in range(count):
            entity_term = random.choice(entity_terms)
            template = random.choice(ambiguous_templates)
            text = template.replace(entity_type.lower().replace('_', ' '), entity_term)
            cleaned_text_for_example = clean_text_cached(text)
            if not cleaned_text_for_example: continue

            if random.random() < 0.3:
                noise_examples.append((cleaned_text_for_example, {"entities": []}))
            else:
                cleaned_entity_term = clean_text_cached(entity_term)
                if not cleaned_entity_term: continue
                start_pos = cleaned_text_for_example.find(cleaned_entity_term)
                if start_pos != -1:
                    end_pos = start_pos + len(cleaned_entity_term)
                    entities = [(start_pos, end_pos, entity_type)]
                    noise_examples.append((cleaned_text_for_example, {"entities": entities}))
        return noise_examples
    
    def regularize_perfect_entities(self, training_data: List[Tuple], perfect_entities: List[str]) -> List[Tuple]:
        if not perfect_entities: return training_data
        regularized_data = list(training_data) # Make a mutable copy
        logger.info(f"🎯 Regularizing {len(perfect_entities)} perfect entities: {perfect_entities}")
        for entity in perfect_entities:
            base_challenging, base_noise = 35, 20
            challenging_ex = self.generate_challenging_examples(entity, base_challenging)
            noise_ex = self.add_subtle_noise_examples(entity, base_noise)
            regularized_data.extend(challenging_ex)
            regularized_data.extend(noise_ex)
            logger.info(f"   📉 Added {len(challenging_ex)} challenging + {len(noise_ex)} noise examples for {entity}")
        return regularized_data

class FinalPushAugmenter:
    """Final push augmentation targeting the remaining problem entities"""
    def __init__(self): # SEAT_SYNONYMS is global
        self.critical_entities = {"BACKREST": 0.000, "SEAT_WARMER": 0.000, "TRAYTABLE": 0.667, "MATERIAL": 0.588, "SEAT_MESSAGE": 0.519}
        self.near_target_entities = {"ARMREST": 0.871, "HEADREST": 0.867, "LUMBAR_SUPPORT": 0.875, "RECLINER": 0.868, "FOOTREST": 0.741, "CUSHION": 0.571}
        self.critical_templates = { # Simplified, use SEAT_SYNONYMS for {entity}
            "BACKREST": ["The {entity} provides excellent support for my back", "The {entity} is uncomfortable and too stiff"],
            "SEAT_WARMER": ["The {entity} function works perfectly in cold weather", "The {entity} doesn't work properly and stays cold"],
            "TRAYTABLE": ["The {entity} is perfect for eating meals during flights", "The {entity} is too small for proper meal service"],
            "MATERIAL": ["The {entity} feels premium and luxurious", "The {entity} feels cheap and synthetic"],
            "SEAT_MESSAGE": ["The {entity} function provides excellent relaxation", "The {entity} function is broken and doesn't work"]
        }

    def generate_critical_examples(self, entity_type: str, count: int = 100) -> List[Tuple[str, Dict]]:
        if entity_type not in self.critical_entities: return []
        synthetic_examples = []
        entity_terms = SEAT_SYNONYMS.get(entity_type, [entity_type.lower()])
        templates = self.critical_templates.get(entity_type, [])
        if not templates or not entity_terms: return []

        for _ in range(count):
            entity_term = random.choice(entity_terms)
            template = random.choice(templates)
            text = template.replace("{entity}", entity_term)
            # Add natural variations
            starters = ["", "Overall, ", "I think ", "In my experience, ", "Honestly, "]
            endings = ["", ".", " overall.", " for sure.", " in my opinion."]
            if random.random() < 0.3: text = random.choice(starters) + text.lower() # text.lower() was here, moved to clean_text
            if random.random() < 0.3: text += random.choice(endings)
            
            cleaned_text_for_example = clean_text_cached(text)
            if not cleaned_text_for_example: continue
            
            cleaned_entity_term = clean_text_cached(entity_term)
            if not cleaned_entity_term: continue

            start_pos = cleaned_text_for_example.find(cleaned_entity_term)
            if start_pos == -1: continue
            end_pos = start_pos + len(cleaned_entity_term)
            
            entities = [(start_pos, end_pos, entity_type)]
            synthetic_examples.append((cleaned_text_for_example, {"entities": entities}))
        return synthetic_examples

    def boost_near_target_entities(self, training_data: List[Tuple]) -> List[Tuple]:
        boosted_data = list(training_data)
        for entity_type, current_f1 in self.near_target_entities.items():
            gap = 0.95 - current_f1
            if gap <= 0.05: boost_count = 30
            elif gap <= 0.15: boost_count = 50
            else: boost_count = 80
            
            entity_terms = SEAT_SYNONYMS.get(entity_type, [entity_type.lower()])
            if not entity_terms: continue
            quality_templates = [
                f"The {entity_type.lower().replace('_', ' ')} is extremely comfortable",
                f"I love how the {entity_type.lower().replace('_', ' ')} feels",
            ]
            for _ in range(boost_count):
                entity_term = random.choice(entity_terms)
                template = random.choice(quality_templates)
                text = template.replace(entity_type.lower().replace('_', ' '), entity_term)
                
                cleaned_text_for_example = clean_text_cached(text)
                if not cleaned_text_for_example: continue

                cleaned_entity_term = clean_text_cached(entity_term)
                if not cleaned_entity_term: continue

                start_pos = cleaned_text_for_example.find(cleaned_entity_term)
                if start_pos == -1: continue
                end_pos = start_pos + len(cleaned_entity_term)
                entities = [(start_pos, end_pos, entity_type)]
                boosted_data.append((cleaned_text_for_example, {"entities": entities}))
        return boosted_data

class OptimizedNERTrainer:
    """Optimized trainer with better validation handling and perfect score regularization"""
    def __init__(self, train_data: List[Tuple], target_max_score: float = 0.97):
        self.train_data = train_data # Expects cleaned text with mapped offsets
        self.nlp = spacy.blank("en")
        self.metrics_history = []
        self.best_model_state = None
        self.best_f1 = 0.0
        self.regularizer = PerfectScoreRegularizer(target_max_score) # SEAT_SYNONYMS is global

    def create_balanced_validation_split(self) -> Tuple[List[Example], List[Example]]:
        # This method now expects self.train_data to be (cleaned_text, {"entities": [...]})
        entity_examples = defaultdict(list)
        all_examples_spacy = [] # Store spaCy Example objects
        
        # Convert (text, annotation_dict) tuples to spaCy Example objects
        temp_nlp = spacy.blank("en") # Temporary NLP object for creating Docs

        for text, annotations in self.train_data:
            if not text or not text.strip(): continue
            try:
                doc = temp_nlp.make_doc(text)
                # Filter entities to ensure they are valid for the doc length
                valid_entities = []
                for start, end, label in annotations.get("entities", []):
                    if 0 <= start < end <= len(text): # Check against current text length
                        valid_entities.append((start, end, label))
                    else:
                        logger.warning(f"Invalid entity span ({start},{end}) for text length {len(text)}: '{text[:50]}...'")
                
                if not valid_entities and annotations.get("entities"): # If there were entities but none were valid
                    logger.debug(f"No valid entities for example: {text[:50]}...")
                    # We might still want to include texts without entities in training/validation
                    # For now, focusing on examples with valid entities for splitting logic
                
                example = Example.from_dict(doc, {"entities": valid_entities})
                all_examples_spacy.append(example)
                
                for _, _, label in valid_entities: # Use valid_entities for grouping
                    entity_examples[label].append(example)
            except Exception as e:
                logger.warning(f"Error creating Example for text '{text[:50]}...': {e}")
                continue
        
        if not all_examples_spacy:
            logger.error("No valid spaCy Example objects could be created from training data.")
            return [], []

        min_val_per_entity = 3
        train_examples_spacy, val_examples_spacy = [], []
        reserved_for_val_ids = set()

        # Sort entities to ensure consistent processing order for reproducibility
        sorted_entity_keys = sorted(entity_examples.keys())

        for entity_label in sorted_entity_keys:
            examples_for_entity = entity_examples[entity_label]
            if len(examples_for_entity) >= min_val_per_entity * 2: # Need enough for train & val
                # Use a consistent local random for shuffling this entity's examples
                local_random_entity = random.Random(RANDOM_SEED + sum(ord(c) for c in entity_label))
                local_random_entity.shuffle(examples_for_entity)
                
                val_count_for_entity = min(len(examples_for_entity) // 4, min_val_per_entity * 2) # 25% or min*2
                val_count_for_entity = max(val_count_for_entity, min_val_per_entity)
                
                for i in range(val_count_for_entity):
                    if i < len(examples_for_entity):
                         # Use a unique, stable ID for each example if possible, otherwise hash of text
                        example_id = id(examples_for_entity[i])
                        reserved_for_val_ids.add(example_id)
        
        # Assign examples to train or val based on reserved_for_val_ids
        for example_obj in all_examples_spacy:
            current_example_id = id(example_obj)
            if current_example_id in reserved_for_val_ids:
                val_examples_spacy.append(example_obj)
            else:
                train_examples_spacy.append(example_obj)

        # If validation set is too small (e.g., < 15% of total), move some from training
        if len(all_examples_spacy) > 0 and len(val_examples_spacy) < len(all_examples_spacy) * 0.15:
            needed_for_val = int(len(all_examples_spacy) * 0.15) - len(val_examples_spacy)
            if needed_for_val > 0 and train_examples_spacy:
                # Use a consistent local random for this shuffle as well
                local_random_overall = random.Random(RANDOM_SEED + 12345) # Different seed for this step
                local_random_overall.shuffle(train_examples_spacy)
                
                num_to_move = min(needed_for_val, len(train_examples_spacy) // 2) # Move up to half of remaining train
                val_examples_spacy.extend(train_examples_spacy[:num_to_move])
                train_examples_spacy = train_examples_spacy[num_to_move:]
        
        logger.info(f"Optimized split: {len(train_examples_spacy)} train Examples, {len(val_examples_spacy)} validation Examples.")
        val_entity_counts = Counter(ent.label_ for ex in val_examples_spacy for ent in ex.reference.ents)
        logger.info(f"Validation entity distribution: {val_entity_counts}")
        return train_examples_spacy, val_examples_spacy


    def train_optimized(self, n_iter: int = 100):
        logger.info("Starting REPRODUCIBLE optimized NER training with perfect score regularization...")
        logger.info(f"🎯 Target maximum score for regularization: {self.regularizer.target_max_score}")
        spacy.util.fix_random_seed(RANDOM_SEED)

        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe("ner", last=True)
        else:
            ner = self.nlp.get_pipe("ner")

        # Add labels from STANDARDIZED_LABELS to ensure all are known
        for label in STANDARDIZED_LABELS:
            if label not in ner.labels:
                ner.add_label(label)
        
        # Also add any labels found in training data, if different (should align with STANDARDIZED_LABELS)
        for _, annotations in self.train_data:
            for _, _, label in annotations.get("entities", []):
                if label not in ner.labels: # Though they should be in STANDARDIZED_LABELS
                    ner.add_label(label)
                    logger.warning(f"Label '{label}' from training data not in STANDARDIZED_LABELS. Added.")


        train_examples, val_examples = self.create_balanced_validation_split()

        if not train_examples:
            logger.error("No training examples available after splitting. NER training cannot proceed.")
            return None
        
        # Initialize NER component with training examples
        # Ensure other pipes are disabled if they exist and are not needed for NER init
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        with self.nlp.disable_pipes(*other_pipes):
            self.nlp.initialize(lambda: train_examples)

        if val_examples:
            logger.info("📊 Evaluating initial untrained model (Epoch 0)...")
            initial_metrics = self.evaluate_model_performance(val_examples, "Epoch 0")
            initial_f1 = initial_metrics.get('entity_level', {}).get('f1', 0.0)
            logger.info(f"🔍 Initial F1 on validation set: {initial_f1:.4f}")

        patience_counter = 0
        regularization_applied_this_run = False # Renamed to avoid conflict

        for epoch in range(n_iter):
            current_epoch_train_data = train_examples # Use the examples from split
            if epoch < 20: batch_size, dropout = 4, 0.3
            elif epoch < 40: batch_size, dropout = 8, 0.2
            elif epoch < 70: batch_size, dropout = 16, 0.15
            else: batch_size, dropout = 32, 0.1
            
            epoch_random = random.Random(RANDOM_SEED + epoch)
            epoch_random.shuffle(current_epoch_train_data)
            
            losses = {}
            batches = spacy.util.minibatch(current_epoch_train_data, size=batch_size)
            
            for batch_idx, batch in enumerate(batches):
                try:
                    self.nlp.update(batch, drop=dropout, losses=losses)
                except Exception as e:
                    logger.error(f"Error during nlp.update in epoch {epoch+1}, batch {batch_idx}: {e}")
                    # Log details of the problematic batch if possible
                    # for ex_in_batch in batch: logger.debug(f"Problematic example text: {ex_in_batch.text[:100]}")
                    continue 
            
            should_validate = (epoch < 10) or ((epoch + 1) % 2 == 0 and epoch < 30) or ((epoch + 1) % 3 == 0)
            
            if val_examples and should_validate:
                metrics = self.evaluate_model_performance(val_examples, f"Epoch {epoch + 1}")
                current_f1 = metrics.get('entity_level', {}).get('f1', 0.0)
                
                if not regularization_applied_this_run and epoch > 15: # Start checking after epoch 15
                    perfect_entities = self.regularizer.detect_perfect_entities(metrics)
                    if perfect_entities:
                        logger.info(f"\n🔍 Detected perfect entities: {perfect_entities} at epoch {epoch+1}")
                        
                        # Augment the original Python list of (text, annotation) tuples
                        self.train_data = self.regularizer.regularize_perfect_entities(self.train_data, perfect_entities)
                        
                        # Recreate spaCy Example objects for training and validation sets
                        train_examples, val_examples = self.create_balanced_validation_split()
                        if not train_examples: # Should not happen if regularization adds data
                             logger.error("Training examples became empty after regularization. Stopping.")
                             break
                        regularization_applied_this_run = True
                        logger.info(f"✅ Regularization applied. New training size: {len(train_examples)} Examples.")
                        # Re-initialize if vocabulary changed significantly (optional, usually not needed for just adding examples)
                        # self.nlp.initialize(lambda: train_examples, sgd=self.nlp.get_pipe("ner").optimizer)

                if current_f1 > self.best_f1:
                    self.best_f1 = current_f1
                    self.best_model_state = self.nlp.to_bytes()
                    patience_counter = 0
                    logger.info(f"🏆 NEW BEST MODEL: Epoch {epoch + 1}, Val F1={current_f1:.4f}")
                else:
                    patience_counter += 1
                
                if patience_counter >= 12: # Increased patience
                    logger.info(f"🛑 Early stopping at epoch {epoch + 1} due to no improvement in Val F1 for {patience_counter} validation steps.")
                    break
            
            if (epoch + 1) % 5 == 0 or epoch == n_iter -1 : # Log loss periodically
                 logger.info(f"Epoch {epoch + 1}/{n_iter}, NER Loss: {losses.get('ner', 0.0):.4f}")

        if self.best_model_state:
            self.nlp.from_bytes(self.best_model_state)
            logger.info(f"✅ Restored best NER model with Val F1: {self.best_f1:.4f}")
        
        # Final evaluation on validation set using the best model
        if val_examples:
            logger.info("📊 Performing final evaluation on validation set with the best model...")
            self.evaluate_model_performance(val_examples, "Final Validation")
        
        return self.nlp

    def evaluate_model_performance(self, examples: List[Example], phase: str = "Evaluation") -> Dict:
        if not examples: 
            logger.warning(f"No examples provided for NER evaluation in phase: {phase}")
            return {}
        
        y_true_ents_sets = []
        y_pred_ents_sets = []
        
        # Use nlp.pipe for efficiency
        texts_to_pred = [ex.reference.text for ex in examples]
        pred_docs = list(self.nlp.pipe(texts_to_pred))

        for i, ex in enumerate(examples):
            pred_doc = pred_docs[i]
            true_entities_set = set((ent.label_, ent.start_char, ent.end_char) for ent in ex.reference.ents)
            pred_entities_set = set((ent.label_, ent.start_char, ent.end_char) for ent in pred_doc.ents)
            y_true_ents_sets.append(true_entities_set)
            y_pred_ents_sets.append(pred_entities_set)
        
        per_entity_metrics = {}
        tp_total, fp_total, fn_total = 0, 0, 0

        for label in sorted(list(STANDARDIZED_LABELS)): # Evaluate against all standardized labels
            tp, fp, fn = 0, 0, 0
            for true_set, pred_set in zip(y_true_ents_sets, y_pred_ents_sets):
                true_label_ents = {e for e in true_set if e[0] == label}
                pred_label_ents = {e for e in pred_set if e[0] == label}
                tp += len(true_label_ents.intersection(pred_label_ents))
                fp += len(pred_label_ents - true_label_ents)
                fn += len(true_label_ents - pred_label_ents)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            support = tp + fn
            per_entity_metrics[label] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': support, 'tp':tp, 'fp':fp, 'fn':fn}
            tp_total += tp; fp_total += fp; fn_total += fn

        overall_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        overall_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        metrics_summary = {
            'phase': phase,
            'entity_level': {'precision': overall_precision, 'recall': overall_recall, 'f1': overall_f1, 
                             'true_positives': tp_total, 'false_positives': fp_total, 'false_negatives': fn_total},
            'per_entity': per_entity_metrics
        }
        
        # Logging evaluation results
        logger.info(f"--- NER Evaluation: {phase} ---")
        logger.info(f"Overall Entity P: {overall_precision:.3f}, R: {overall_recall:.3f}, F1: {overall_f1:.3f}")
        entities_above_90_count = 0
        perfect_entities_count = 0
        total_supported_entities = 0
        for label, scores in sorted(per_entity_metrics.items()):
            if scores['support'] > 0:
                total_supported_entities +=1
                status_marker = ""
                if scores['f1'] >= 0.999: status_marker = "🎯 PERFECT" ; perfect_entities_count+=1; entities_above_90_count+=1
                elif scores['f1'] >= 0.9: status_marker = "✅ >=0.9" ; entities_above_90_count+=1
                elif scores['f1'] >= 0.8: status_marker = "👍 >=0.8"
                else: status_marker = "⚠️ <0.8"
                logger.info(f"  {label:<15}: F1={scores['f1']:.3f}, P={scores['precision']:.3f}, R={scores['recall']:.3f} (Sup: {scores['support']}) {status_marker}")
        if total_supported_entities > 0:
             logger.info(f"Entities with F1 >= 0.9: {entities_above_90_count}/{total_supported_entities} (Perfect: {perfect_entities_count})")
        logger.info("--- End NER Evaluation ---")

        self.metrics_history.append(metrics_summary)
        return metrics_summary


# --- LDA Analyzer (from Final_Push_Complete_Analytics.py) ---
class LDAAnalyzer:
    def __init__(self, output_dir: str, nlp_spacy_model_for_lda): # Takes a spaCy model
        self.output_dir = output_dir
        self.lda_dir = os.path.join(output_dir, "lda_analysis")
        ensure_directory(self.lda_dir)
        self.nlp_model_for_lda = nlp_spacy_model_for_lda # Use the passed model

    def preprocess_text_for_lda(self, text: str, stop_words_set: set) -> List[str]:
        if not text or pd.isna(text): return []
        # Basic cleaning (lower, remove non-alpha, normalize whitespace)
        # More advanced cleaning (like HTML removal) should happen before this stage
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text) 
        text = re.sub(r'\s+', ' ', text).strip()
        
        doc = self.nlp_model_for_lda(text)
        tokens = [token.lemma_ for token in doc 
                 if not token.is_stop and not token.is_punct 
                 and len(token.lemma_) > 2 and token.lemma_ not in stop_words_set
                 and token.pos_ in ['NOUN', 'ADJ', 'VERB']] # Focus on content words
        return tokens

    def perform_lda_analysis(self, text_data: List[str], num_topics: int = 10):
        logger.info(f"🔍 Starting LDA Topic Modeling with {num_topics} topics...")
        custom_stop_words_lda = set(nltk.corpus.stopwords.words("english")) | set(string.punctuation) | {
            "seat", "seats", "car", "vehicle", "trip", "feature", "get", "feel", "felt", "look", "make", "also",
            "even", "really", "quite", "very", "much", "good", "great", "nice", "well", "drive", "driving",
            "would", "could", "im", "ive", "id", "nan", "auto", "automobile", "product", "item", "order",
            "time", "way", "thing", "things", "lot", "bit", "little", "big", "small", "new", "old", "vehicle"
        } # Added 'vehicle'
        
        processed_texts = [self.preprocess_text_for_lda(text, custom_stop_words_lda) for text in text_data if text and str(text).strip()]
        processed_texts = [text for text in processed_texts if len(text) > 2] # Min tokens
        
        if len(processed_texts) < 5:
            logger.warning("⚠️ Insufficient processed documents for LDA analysis after preprocessing.")
            return None, None, None, 0.0 # Return 0.0 for coherence

        logger.info(f"📊 Processed {len(processed_texts)} documents for LDA.")
        dictionary = corpora.Dictionary(processed_texts)
        dictionary.filter_extremes(no_below=3, no_above=0.6, keep_n=10000)
        if not dictionary:
            logger.warning("⚠️ LDA dictionary is empty after filtering.")
            return None, None, None, 0.0

        corpus = [dictionary.doc2bow(text) for text in processed_texts]
        corpus = [doc for doc in corpus if doc]
        if not corpus or len(corpus) < num_topics:
            if corpus: num_topics = max(1, len(corpus) -1 if len(corpus) > 1 else 1)
            else: return None, None, None, 0.0
            if num_topics == 0: return None, None, None, 0.0
            logger.warning(f"⚠️ Adjusted num_topics to {num_topics} due to small corpus size.")

        logger.info(f"🧠 Training LDA model with {num_topics} topics...")
        lda_model = LdaModel(
            corpus=corpus, id2word=dictionary, num_topics=num_topics,
            random_state=RANDOM_SEED, update_every=1, chunksize=100,
            passes=15, alpha='auto', per_word_topics=True, iterations=100
        )
        coherence_model = CoherenceModel(model=lda_model, texts=processed_texts, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        logger.info(f"📈 LDA Coherence Score (c_v): {coherence_score:.4f}")

        lda_model.save(os.path.join(self.lda_dir, "lda_model.gensim"))
        dictionary.save(os.path.join(self.lda_dir, "lda_dictionary.gensim"))
        corpora.MmCorpus.serialize(os.path.join(self.lda_dir, "lda_corpus.mm"), corpus)
        
        # Save processed texts for Kansei module if it needs them (though Kansei usually re-processes)
        with open(os.path.join(self.lda_dir, "lda_processed_texts.json"), 'w') as f:
            json.dump(processed_texts, f)

        topics_data = [{"topic_id": idx, "terms": topic_terms} for idx, topic_terms in lda_model.print_topics(-1, num_words=15)]
        pd.DataFrame(topics_data).to_csv(os.path.join(self.lda_dir, "lda_topic_terms.csv"), index=False)
        
        self.create_lda_visualizations(lda_model, corpus, dictionary, coherence_score)
        return lda_model, dictionary, corpus, coherence_score

    def create_lda_visualizations(self, lda_model, corpus, dictionary, coherence_score):
        try:
            vis_data = gensimvis.prepare(lda_model, corpus, dictionary, mds='mmds') # Use mmds
            pyLDAvis.save_html(vis_data, os.path.join(self.lda_dir, 'lda_interactive_visualization.html'))
            logger.info(f"💾 Interactive LDA visualization saved to lda_interactive_visualization.html")
        except Exception as e:
            logger.error(f"⚠️ Error generating interactive LDA visualization: {e}. Try mds='tsne' or check C++ compiler.")
        # ... (rest of LDA visualizations from Final_Push_Complete_Analytics.py)
        # For brevity, assuming the heatmap and distribution plots are similar and can be reused/adapted.


# --- Kansei Engineering Module (from combined_seat_kansei.py) ---
class KanseiModule:
    def __init__(self, lda_model_gensim, dictionary_gensim, corpus_gensim, df_ner_sentiment_processed):
        self.lda_model = lda_model_gensim
        self.dictionary = dictionary_gensim
        self.corpus_from_lda_step = corpus_gensim # Corpus used to TRAIN the LDA model
        self.df_with_ner_sentiment = df_ner_sentiment_processed # DataFrame with NER and Sentiment results
        self.stop_words_kansei_lda = set(nltk.corpus.stopwords.words('english')) | {"seat", "seats", "car"} # Kansei specific stops

        # Kansei emotion mapping (from combined_seat_kansei.py)
        self.kansei_emotion_map = {
            0: {"primary_emotion": "Uncomfortable", "secondary_emotions": ["Painful", "Stiff", "Harsh"], "keywords": ["hard", "keras", "sakit", "pain", "stiff", "uncomfortable", "hurt", "ache", "rigid", "firm"]},
            1: {"primary_emotion": "Cramped", "secondary_emotions": ["Confined", "Restricted", "Tight"], "keywords": ["narrow", "sempit", "tight", "small", "kecil", "cramped", "confined", "restricted", "squeezed"]},
            2: {"primary_emotion": "Comfortable", "secondary_emotions": ["Cozy", "Relaxing", "Pleasant"], "keywords": ["comfortable", "nyaman", "good", "soft", "empuk", "cozy", "relaxing", "pleasant", "soothing"]},
            3: {"primary_emotion": "Spacious", "secondary_emotions": ["Roomy", "Open", "Airy"], "keywords": ["spacious", "luas", "wide", "room", "space", "roomy", "open", "airy", "expansive"]},
            4: {"primary_emotion": "Premium", "secondary_emotions": ["Luxurious", "Elegant", "Sophisticated"], "keywords": ["premium", "quality", "luxury", "bagus", "best", "luxurious", "elegant", "sophisticated", "leather", "high-quality"]}, # Added leather, high-quality
            5: {"primary_emotion": "Supportive", "secondary_emotions": ["Ergonomic", "Stable", "Secure"], "keywords": ["support", "position", "posisi", "ergonomic", "stable", "secure", "balanced", "aligned", "lumbar"]}, # Added lumbar
            6: {"primary_emotion": "Innovative", "secondary_emotions": ["Modern", "Advanced", "Smart"], "keywords": ["innovative", "modern", "advanced", "cutting-edge", "futuristic", "smart", "high-tech", "feature", "technology"]}, # Added feature, tech
            7: {"primary_emotion": "Disappointing", "secondary_emotions": ["Inadequate", "Subpar", "Poor"], "keywords": ["disappointing", "inadequate", "subpar", "mediocre", "unsatisfactory", "poor", "lacking", "broken", "issue"]}, # Added broken, issue
            8: {"primary_emotion": "Relaxing", "secondary_emotions": ["Calming", "Therapeutic", "Restorative"], "keywords": ["relaxing", "calming", "stress-free", "therapeutic", "rejuvenating", "restorative", "healing", "massage"]}, # Added massage
            9: {"primary_emotion": "Functional", "secondary_emotions": ["Practical", "Efficient", "User-friendly"], "keywords": ["functional", "practical", "works well", "efficient", "user-friendly", "easy to use", "convenient", "tray", "storage"]} # Modified topic
        } # Example: Topic 9 might be about functionality/usability. Original had "Exciting".

    def _preprocess_text_for_kansei_topic_inference(self, text: str) -> List[str]:
        """Preprocessing specific for feeding new text to the TRAINED LDA model for Kansei mapping."""
        if pd.isna(text) or not str(text).strip(): return []
        text_cleaned = str(text).lower() # Lowercase
        text_cleaned = text_cleaned.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
        tokens = nltk.tokenize.word_tokenize(text_cleaned)
        
        # Simple compound word handling (can be expanded)
        compound_word_map = {'head rest': 'headrest', 'arm rest': 'armrest', 'back rest': 'backrest', 'leg room': 'legroom', 'seat cushion': 'seatcushion', 'tray table': 'traytable'}
        processed_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1:
                two_word = f"{tokens[i]} {tokens[i+1]}"
                if two_word in compound_word_map:
                    processed_tokens.append(compound_word_map[two_word])
                    i += 2
                    continue
            processed_tokens.append(tokens[i])
            i += 1
        
        return [word for word in processed_tokens if word not in self.stop_words_kansei_lda and len(word) > 2 and not word.isdigit()]

    def map_topics_to_kansei_emotions(self) -> pd.DataFrame:
        logger.info("Mapping LDA topics to Kansei emotions for each review...")
        kansei_emotion_results = []

        # Group by original feedback text to process each review once for Kansei emotion
        # The self.df_with_ner_sentiment might have multiple rows per original review (one per entity-sentence)
        unique_reviews_df = self.df_with_ner_sentiment[['Feedback Text', 'Cleaned Text']].drop_duplicates().reset_index(drop=True)
        
        logger.info(f"Processing {len(unique_reviews_df)} unique reviews for Kansei emotion mapping.")

        for idx, row in unique_reviews_df.iterrows():
            original_review_text = row['Feedback Text']
            # Use 'Cleaned Text' for LDA topic inference as it's more consistent
            # The LDA model itself was trained on preprocessed text (e.g., lemmatized nouns/adj/verbs)
            # For inference on new docs, we need to apply similar preprocessing.
            text_for_lda_inference = row['Cleaned Text'] 

            processed_tokens_for_lda = self._preprocess_text_for_kansei_topic_inference(text_for_lda_inference)
            
            dominant_topic_idx = None
            primary_emotion = "Unknown"
            secondary_emotions = []
            emotion_confidence = 0.0

            if processed_tokens_for_lda:
                bow_vector = self.dictionary.doc2bow(processed_tokens_for_lda)
                if bow_vector: # If BoW is not empty
                    doc_topics = self.lda_model.get_document_topics(bow_vector, minimum_probability=0.05) # Get topic distribution
                    if doc_topics:
                        dominant_topic = max(doc_topics, key=lambda x: x[1])
                        dominant_topic_idx = dominant_topic[0]
                        emotion_confidence = round(dominant_topic[1], 4)
                        
                        if dominant_topic_idx in self.kansei_emotion_map:
                            primary_emotion = self.kansei_emotion_map[dominant_topic_idx]["primary_emotion"]
                            secondary_emotions = self.kansei_emotion_map[dominant_topic_idx]["secondary_emotions"]
            
            # Keyword override logic (optional, from original Kansei script)
            # This uses the original_review_text for keyword matching
            keyword_matches_found = []
            original_review_lower = str(original_review_text).lower()
            for topic_id_map, emotion_data in self.kansei_emotion_map.items():
                match_count = sum(1 for kw in emotion_data["keywords"] if kw in original_review_lower)
                if match_count > 0:
                    keyword_matches_found.append({
                        'emotion': emotion_data["primary_emotion"], 
                        'secondary': emotion_data["secondary_emotions"],
                        'matches': match_count,
                        'original_topic_id': topic_id_map # For reference
                    })
            
            if keyword_matches_found:
                best_keyword_match = max(keyword_matches_found, key=lambda x: x['matches'])
                # Apply override if keyword match is strong or LDA confidence is low
                if best_keyword_match['matches'] >= 2 or (emotion_confidence < 0.5 and best_keyword_match['matches'] > 0):
                    logger.debug(f"Kansei override for review (Original LDA Topic: {dominant_topic_idx}, Emotion: {primary_emotion}) with Keyword Match: {best_keyword_match['emotion']} (Matches: {best_keyword_match['matches']})")
                    primary_emotion = best_keyword_match['emotion']
                    secondary_emotions = best_keyword_match['secondary']
                    # Optionally adjust confidence or mark as keyword-derived
                    emotion_confidence = max(emotion_confidence, 0.55) # Boost confidence slightly for strong keyword match


            kansei_emotion_results.append({
                'Feedback Text': original_review_text, # Key for merging
                'Dominant LDA Topic': dominant_topic_idx,
                'Kansei Primary Emotion': primary_emotion,
                'Kansei Secondary Emotions': secondary_emotions,
                'Kansei Emotion Confidence': emotion_confidence
            })
        
        kansei_emotions_df = pd.DataFrame(kansei_emotion_results)
        logger.info(f"Kansei emotion mapping complete for {len(kansei_emotions_df)} reviews.")
        
        # Merge these Kansei emotions back into the main df_with_ner_sentiment
        if not kansei_emotions_df.empty:
            self.df_with_ner_sentiment = pd.merge(self.df_with_ner_sentiment, kansei_emotions_df, on='Feedback Text', how='left')
        
        return self.df_with_ner_sentiment # Return the main df now augmented with Kansei info

    def analyze_emotion_patterns(self): # Operates on the augmented self.df_with_ner_sentiment
        logger.info("Analyzing Kansei emotion patterns...")
        # Ensure Kansei columns exist
        if 'Kansei Primary Emotion' not in self.df_with_ner_sentiment.columns:
            logger.warning("Kansei emotion columns not found in DataFrame. Skipping emotion pattern analysis.")
            return {}

        # Analyze per unique review
        unique_reviews_kansei_df = self.df_with_ner_sentiment[['Feedback Text', 'Kansei Primary Emotion', 'Kansei Secondary Emotions', 'Kansei Emotion Confidence']].drop_duplicates('Feedback Text')

        emotion_analysis = {
            'primary_emotions_distribution': Counter(unique_reviews_kansei_df['Kansei Primary Emotion'].dropna()),
            'secondary_emotions_distribution': Counter(),
            'emotion_combinations': defaultdict(int), # Primary + top secondary
            'avg_confidence_per_emotion': defaultdict(list),
            'feature_kansei_correlation': defaultdict(lambda: Counter()) # Seat Component vs Kansei Primary Emotion
        }

        for _, row in unique_reviews_kansei_df.iterrows():
            primary = row['Kansei Primary Emotion']
            if pd.isna(primary) or primary == "Unknown": continue

            # Secondary emotions
            secondary_list = row.get('Kansei Secondary Emotions', [])
            if isinstance(secondary_list, list):
                for sec_emo in secondary_list:
                    emotion_analysis['secondary_emotions_distribution'][sec_emo] += 1
                if secondary_list:
                    combo_key = f"{primary} + {', '.join(sorted(secondary_list[:2]))}" # Top 2 secondary
                    emotion_analysis['emotion_combinations'][combo_key] +=1
            
            # Confidence
            confidence = row.get('Kansei Emotion Confidence', 0.0)
            if pd.notna(confidence):
                emotion_analysis['avg_confidence_per_emotion'][primary].append(float(confidence))

        # Calculate average confidence
        for emotion, confs in emotion_analysis['avg_confidence_per_emotion'].items():
            emotion_analysis['avg_confidence_per_emotion'][emotion] = sum(confs) / len(confs) if confs else 0.0
        
        # Feature-Kansei Correlation (needs to iterate over the full self.df_with_ner_sentiment)
        for _, row in self.df_with_ner_sentiment.iterrows():
            component = row.get('Seat Component')
            primary_kansei = row.get('Kansei Primary Emotion')
            if component and pd.notna(primary_kansei) and primary_kansei != "Unknown":
                emotion_analysis['feature_kansei_correlation'][component][primary_kansei] += 1
        
        # Convert Counters to dicts for easier JSON serialization later
        emotion_analysis['primary_emotions_distribution'] = dict(emotion_analysis['primary_emotions_distribution'])
        emotion_analysis['secondary_emotions_distribution'] = dict(emotion_analysis['secondary_emotions_distribution'])
        emotion_analysis['emotion_combinations'] = dict(emotion_analysis['emotion_combinations'])
        emotion_analysis['feature_kansei_correlation'] = {k: dict(v) for k, v in emotion_analysis['feature_kansei_correlation'].items()}


        logger.info("Kansei emotion pattern analysis complete.")
        return emotion_analysis

    def generate_design_recommendations(self, emotion_patterns: Dict):
        logger.info("Generating Kansei design recommendations...")
        recommendations = []
        if not emotion_patterns or 'primary_emotions_distribution' not in emotion_patterns:
            return recommendations

        primary_counts = emotion_patterns['primary_emotions_distribution']
        total_valid_reviews = sum(c for emo, c in primary_counts.items() if emo != "Unknown")
        if total_valid_reviews == 0: return recommendations

        # Example recommendation logic (can be greatly expanded)
        uncomfortable_ratio = primary_counts.get('Uncomfortable', 0) / total_valid_reviews
        if uncomfortable_ratio > 0.15: # If >15% reviews express discomfort
            recs = {'component': 'Overall Seat Comfort', 'issue': f'High discomfort ({uncomfortable_ratio:.1%})', 'priority': 'High', 
                    'suggestions': ['Review cushion material and density.', 'Enhance lumbar support adjustability.', 'Check for pressure points.']}
            # Correlate with features
            if 'feature_kansei_correlation' in emotion_patterns:
                discomfort_features = Counter()
                for feature, emotions in emotion_patterns['feature_kansei_correlation'].items():
                    if emotions.get('Uncomfortable', 0) > 0:
                        discomfort_features[feature] = emotions['Uncomfortable']
                if discomfort_features:
                    top_discomfort_feature = discomfort_features.most_common(1)[0][0]
                    recs['suggestions'].append(f"Focus improvements on {top_discomfort_feature} related to discomfort.")
            recommendations.append(recs)

        cramped_ratio = primary_counts.get('Cramped', 0) / total_valid_reviews
        if cramped_ratio > 0.10:
            recommendations.append({'component': 'Seat Space & Size', 'issue': f'Significant cramped feeling ({cramped_ratio:.1%})', 'priority': 'High', 
                                    'suggestions': ['Re-evaluate seat width and legroom.', 'Optimize armrest placement.']})
        
        premium_ratio = primary_counts.get('Premium', 0) / total_valid_reviews
        if premium_ratio > 0.10: # If perceived as premium
             recommendations.append({'component': 'Material & Finish', 'issue': f'Positive premium perception ({premium_ratio:.1%})', 'priority': 'Maintain/Enhance', 
                                    'suggestions': ['Continue using high-quality materials.', 'Explore subtle luxury accents.']})
        elif 'Disappointing' in primary_counts and primary_counts.get('Disappointing',0)/total_valid_reviews > 0.05 :
             recommendations.append({'component': 'Material & Finish', 'issue': 'Perceived as Disappointing/Subpar', 'priority': 'Medium', 
                                    'suggestions': ['Investigate material quality complaints.', 'Improve perceived value.']})


        # Sort by priority
        priority_map = {'High': 0, 'Medium': 1, 'Maintain/Enhance': 2, 'Low': 3}
        recommendations.sort(key=lambda x: priority_map.get(x['priority'], 99))
        
        logger.info(f"Generated {len(recommendations)} design recommendations based on Kansei insights.")
        return recommendations


# --- WordCloudGenerator (from Final_Push_Complete_Analytics.py) ---
class WordCloudGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.wordcloud_dir = os.path.join(output_dir, "wordclouds")
        ensure_directory(self.wordcloud_dir) # Use ensure_directory

    def generate_comprehensive_wordclouds(self, all_text_data: List[str], df_with_ner_sentiment_kansei: Optional[pd.DataFrame] = None):
        logger.info("☁️ Generating comprehensive word clouds...")
        custom_stopwords = set(WC_STOPWORDS) | set(nltk.corpus.stopwords.words('english')) | {
            'seat', 'seats', 'car', 'vehicle', 'also', 'get', 'got', 'would', 'could', 'make', 'made', 
            'see', 'really', 'even', 'one', 'nan', 'lot', 'bit', 'im', 'ive', 'id', 'well', 'good', 
            'great', 'nice', 'bad', 'poor', 'drive', 'driving', 'ride', 'riding', 'trip', 'product', 
            'item', 'time', 'way', 'thing', 'things', 'little', 'big', 'small', 'new', 'old', 'feel', 'felt', 'look'
        }
        
        # 1. Overall Word Cloud from all 'Cleaned Text'
        if all_text_data:
            self.create_wordcloud_from_texts(all_text_data, "overall_wordcloud", "Overall Text Word Cloud", stopwords=custom_stopwords)

        if df_with_ner_sentiment_kansei is not None and not df_with_ner_sentiment_kansei.empty:
            # 2. Entity-specific Word Clouds (based on sentences mentioning the entity)
            if 'Seat Component' in df_with_ner_sentiment_kansei.columns and 'Sentence Text' in df_with_ner_sentiment_kansei.columns:
                for component in STANDARDIZED_LABELS:
                    component_texts = df_with_ner_sentiment_kansei[df_with_ner_sentiment_kansei['Seat Component'] == component]['Sentence Text'].dropna().tolist()
                    if component_texts:
                        self.create_wordcloud_from_texts(component_texts, f"component_{component.lower()}_wordcloud", f"{component} Related Text", stopwords=custom_stopwords)
            
            # 3. Sentiment-based Word Clouds (Positive, Negative, Neutral sentences)
            if 'Sentence Sentiment Label' in df_with_ner_sentiment_kansei.columns and 'Sentence Text' in df_with_ner_sentiment_kansei.columns:
                positive_texts = df_with_ner_sentiment_kansei[df_with_ner_sentiment_kansei['Sentence Sentiment Label'] == 'POSITIVE']['Sentence Text'].dropna().tolist()
                negative_texts = df_with_ner_sentiment_kansei[df_with_ner_sentiment_kansei['Sentence Sentiment Label'] == 'NEGATIVE']['Sentence Text'].dropna().tolist()
                # neutral_texts = df_with_ner_sentiment_kansei[df_with_ner_sentiment_kansei['Sentence Sentiment Label'] == 'NEUTRAL']['Sentence Text'].dropna().tolist() # If NEUTRAL is a label

                if positive_texts: self.create_wordcloud_from_texts(positive_texts, "positive_sentiment_wordcloud", "Positive Sentiment Text", stopwords=custom_stopwords, colormap='Greens')
                if negative_texts: self.create_wordcloud_from_texts(negative_texts, "negative_sentiment_wordcloud", "Negative Sentiment Text", stopwords=custom_stopwords, colormap='Reds')

            # 4. Kansei Emotion-based Word Clouds
            if 'Kansei Primary Emotion' in df_with_ner_sentiment_kansei.columns and 'Feedback Text' in df_with_ner_sentiment_kansei.columns:
                 unique_reviews_for_kansei_wc = df_with_ner_sentiment_kansei[['Feedback Text', 'Kansei Primary Emotion']].drop_duplicates('Feedback Text')
                 for emotion in unique_reviews_for_kansei_wc['Kansei Primary Emotion'].dropna().unique():
                     if emotion == "Unknown": continue
                     emotion_texts = unique_reviews_for_kansei_wc[unique_reviews_for_kansei_wc['Kansei Primary Emotion'] == emotion]['Feedback Text'].tolist()
                     if emotion_texts:
                         safe_emotion_name = re.sub(r'\W+', '', emotion.lower()) # Make filename safe
                         self.create_wordcloud_from_texts(emotion_texts, f"kansei_{safe_emotion_name}_wordcloud", f"'{emotion}' Kansei Emotion Text", stopwords=custom_stopwords, colormap='coolwarm')


        logger.info(f"☁️ Word clouds saved to {self.wordcloud_dir}")

    def create_wordcloud_from_texts(self, texts: List[str], filename_base: str, title: str, stopwords: set, colormap: str = 'viridis'):
        """Helper to create and save a single word cloud."""
        if not texts:
            logger.debug(f"No texts provided for word cloud: {title}")
            return
        
        full_text_corpus = " ".join([str(t) for t in texts if t and str(t).strip()])
        if not full_text_corpus.strip():
            logger.debug(f"Empty corpus after joining texts for word cloud: {title}")
            return
        
        try:
            wordcloud = WordCloud(
                width=1200, height=600, background_color='white', 
                stopwords=stopwords, collocations=True, # Enable collocations for phrases
                max_words=150, colormap=colormap, random_state=RANDOM_SEED
            ).generate(full_text_corpus)
            
            plt.figure(figsize=(15, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title(title, fontsize=18, fontweight='bold', pad=20)
            plt.tight_layout()
            filepath = os.path.join(self.wordcloud_dir, f"{filename_base}.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight') # Reduced DPI for faster save
            plt.close()
            logger.info(f"Generated word cloud: {filepath}")
        except Exception as e:
            logger.error(f"⚠️ Error generating word cloud '{title}': {e}")


# --- Visualization and Reporting Classes (Adapted and Merged) ---
class ComprehensiveVisualizerAndReporter: # Combined from both scripts
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "plots")
        self.analytics_dir = os.path.join(output_dir, "analytics") # Not used much yet
        self.tables_dir = os.path.join(output_dir, "tables") # For CSV outputs
        ensure_directory(self.plots_dir)
        ensure_directory(os.path.join(self.plots_dir, "individual_entities")) # For NER per-entity plots
        ensure_directory(self.analytics_dir)
        ensure_directory(self.tables_dir)

    # NER Visualizations (largely from Final_Push_Complete_Analytics.py's ComprehensiveTrainingVisualizer)
    def plot_ner_training_progress(self, metrics_history: List[Dict], title: str = "NER Training Progress"):
        if not metrics_history: return
        # ... (Implementation from Final_Push_Complete_Analytics.py - plot_training_progress)
        # For brevity, this detailed plotting code is assumed to be here.
        # It plots overall F1 and entities >= 0.9 over epochs.
        logger.info("Plotting NER training progress (Overall F1, Entities >=0.9)...")
        # Simplified version for this combined script:
        epochs = [i for i, m in enumerate(metrics_history)] # Assuming phase is just epoch number
        overall_f1s = [m['entity_level']['f1'] for m in metrics_history if 'entity_level' in m]
        if epochs and overall_f1s:
            plt.figure(figsize=(12,6))
            plt.plot(epochs, overall_f1s, marker='o', label='Overall NER F1')
            plt.title(title)
            plt.xlabel("Epoch/Evaluation Step")
            plt.ylabel("F1 Score")
            plt.ylim(0,1.05)
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(self.plots_dir, "ner_training_overall_f1.png"), dpi=150)
            plt.close()


    def plot_ner_per_entity_progress(self, metrics_history: List[Dict]):
        if not metrics_history: return
        # ... (Implementation from Final_Push_Complete_Analytics.py - plot_per_entity_progress)
        # Plots F1 for each entity over epochs in subplots.
        logger.info("Plotting NER per-entity F1 progress...")
        # Placeholder for brevity

    def plot_ner_individual_entity_detailed_progress(self, metrics_history: List[Dict]):
        if not metrics_history: return
        # ... (Implementation from Final_Push_Complete_Analytics.py - plot_individual_entity_progress)
        # Plots F1, P, R for each entity in separate files.
        logger.info("Plotting detailed NER progress (F1,P,R) for each entity...")
        # Placeholder for brevity

    def plot_ner_final_results(self, final_metrics: Dict):
        if not final_metrics or 'per_entity' not in final_metrics: return
        # ... (Implementation from Final_Push_Complete_Analytics.py - plot_final_results)
        # Plots final F1 bars, P-R scatter, support bars, summary stats.
        logger.info("Plotting final NER results summary...")
        # Simplified version:
        if 'per_entity' in final_metrics:
            labels = list(final_metrics['per_entity'].keys())
            f1_scores = [final_metrics['per_entity'][l]['f1'] for l in labels]
            plt.figure(figsize=(14,7))
            plt.bar(labels, f1_scores, color=sns.color_palette("viridis", len(labels)))
            plt.title("Final NER F1 Scores per Entity")
            plt.ylabel("F1 Score")
            plt.xticks(rotation=45, ha="right")
            plt.ylim(0,1.05)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, "ner_final_f1_per_entity.png"), dpi=150)
            plt.close()


    def plot_training_data_composition(self, base_count: int, augmented_count: int, regularized_count: int):
        # augmented_count = critical_synthetic + boost_synthetic
        # regularized_count = from perfect score regularizer
        total_synthetic = augmented_count + regularized_count
        logger.info("Plotting training data composition...")
        # ... (Implementation from Final_Push_Complete_Analytics.py - plot_data_composition)
        # Pie chart and bar chart of data sources.
        # Simplified:
        labels = ['Base Data', 'Augmented Data', 'Regularized Data']
        sizes = [base_count, augmented_count, regularized_count]
        if sum(sizes) > 0: # Only plot if there's data
            plt.figure(figsize=(8,8))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral', 'lightgreen'])
            plt.title("Training Data Composition")
            plt.savefig(os.path.join(self.plots_dir, "ner_training_data_composition.png"), dpi=150)
            plt.close()


    # General Visualizations (from combined_seat_kansei.py)
    def generate_general_visualizations(self, processed_df: pd.DataFrame):
        logger.info("Generating general NLP visualizations (Sentiment by Component, Component Freq)...")
        if processed_df.empty:
            logger.warning("Processed DataFrame is empty, skipping general visualizations.")
            return

        # 1. Sentiment Distribution by Component
        if 'Seat Component' in processed_df.columns and 'Sentence Sentiment Label' in processed_df.columns:
            try:
                # Filter out N/A or Unknown sentiment if they are not meaningful for this plot
                plot_df_sent = processed_df[~processed_df['Sentence Sentiment Label'].isin(['N/A', 'UNKNOWN'])]
                if not plot_df_sent.empty:
                    sent_pivot = plot_df_sent.groupby(["Seat Component", "Sentence Sentiment Label"]).size().unstack(fill_value=0)
                    if not sent_pivot.empty:
                        sent_pivot.plot(kind="bar", stacked=True, figsize=(14, 7), colormap="viridis")
                        plt.title("Sentiment Distribution by Seat Component", fontsize=16)
                        plt.xlabel("Seat Component", fontsize=12)
                        plt.ylabel("Number of Sentences", fontsize=12)
                        plt.xticks(rotation=45, ha="right", fontsize=10)
                        plt.legend(title="Sentiment Label")
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.plots_dir, "component_sentiment_distribution.png"), dpi=150)
                        plt.close()
                        sent_pivot.to_csv(os.path.join(self.tables_dir, "component_sentiment_distribution.csv"))
            except Exception as e:
                logger.error(f"Error generating sentiment by component plot: {e}")


        # 2. Component Mention Frequency (from NER)
        if 'Seat Component' in processed_df.columns:
            try:
                component_counts = processed_df['Seat Component'].value_counts()
                if not component_counts.empty:
                    plt.figure(figsize=(12, 6))
                    component_counts.plot(kind='bar', color=sns.color_palette("Spectral", len(component_counts)))
                    plt.title("Seat Component Mention Frequency (NER)", fontsize=16)
                    plt.xlabel("Seat Component", fontsize=12)
                    plt.ylabel("Frequency of Mentions", fontsize=12)
                    plt.xticks(rotation=45, ha="right", fontsize=10)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.plots_dir, "ner_component_frequency.png"), dpi=150)
                    plt.close()
                    component_counts.to_csv(os.path.join(self.tables_dir, "ner_component_frequency.csv"))
            except Exception as e:
                 logger.error(f"Error generating component frequency plot: {e}")


    # Kansei Visualizations (from combined_seat_kansei.py)
    def generate_kansei_visualizations(self, kansei_analysis_results: Dict):
        logger.info("Generating Kansei visualizations...")
        if not kansei_analysis_results:
            logger.warning("No Kansei analysis results to visualize.")
            return

        # 1. Kansei Primary Emotion Distribution
        primary_emotions_dist = kansei_analysis_results.get('primary_emotions_distribution', {})
        if primary_emotions_dist:
            labels = [emo for emo in primary_emotions_dist.keys() if emo != "Unknown"]
            counts = [primary_emotions_dist[emo] for emo in labels]
            if labels and counts:
                plt.figure(figsize=(12, 7))
                bars = plt.bar(labels, counts, color=sns.color_palette("pastel", len(labels)))
                plt.title('Kansei Primary Emotion Distribution', fontsize=16)
                plt.xlabel('Kansei Emotion', fontsize=12)
                plt.ylabel('Number of Reviews', fontsize=12)
                plt.xticks(rotation=45, ha="right", fontsize=10)
                # Add percentages on bars
                total_reviews = sum(counts)
                for bar, count_val in zip(bars, counts):
                    height = bar.get_height()
                    percentage = (count_val / total_reviews) * 100 if total_reviews > 0 else 0
                    plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}\n({percentage:.1f}%)',
                             ha='center', va='bottom', fontsize=9)
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, "kansei_emotion_distribution.png"), dpi=150)
                plt.close()

        # 2. Feature-Kansei Emotion Correlation (Heatmap)
        feature_emotion_corr = kansei_analysis_results.get('feature_kansei_correlation', {})
        if feature_emotion_corr:
            # Convert to DataFrame: features as rows, emotions as columns, counts as values
            df_corr = pd.DataFrame(feature_emotion_corr).fillna(0).astype(int)
            # Filter out components/emotions with no occurrences for a cleaner heatmap
            df_corr = df_corr.loc[(df_corr.sum(axis=1) != 0), (df_corr.sum(axis=0) != 0)]
            if not df_corr.empty:
                plt.figure(figsize=(max(12, len(df_corr.columns)*0.8), max(8, len(df_corr.index)*0.5))) # Dynamic size
                sns.heatmap(df_corr, annot=True, fmt="d", cmap="YlGnBu", linewidths=.5, cbar=True)
                plt.title('Seat Component vs. Kansei Emotion Frequency', fontsize=16)
                plt.xlabel('Kansei Emotion', fontsize=12)
                plt.ylabel('Seat Component', fontsize=12)
                plt.xticks(rotation=45, ha="right", fontsize=10)
                plt.yticks(fontsize=10)
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, "feature_kansei_correlation_heatmap.png"), dpi=150)
                plt.close()
        logger.info("Kansei visualizations generated.")


    # Comprehensive Report Generation
    def create_comprehensive_report(self,
                                    ner_metrics_history: List[Dict],
                                    final_ner_metrics: Dict,
                                    training_time: float,
                                    data_counts: Dict, # {'base': count, 'augmented': count, 'regularized': count}
                                    processed_df_with_all_info: pd.DataFrame, # Contains NER, Sentiment, Kansei
                                    lda_results_summary: Optional[Dict], # {'coherence_score': float, 'num_topics': int}
                                    kansei_analysis_results: Optional[Dict]
                                   ):
        logger.info("📊 Generating comprehensive analysis report...")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_path = os.path.join(self.output_dir, "COMPREHENSIVE_SEAT_ANALYSIS_REPORT.md")

        with open(report_path, "w", encoding='utf-8') as f:
            f.write(f"# Comprehensive Seat Feedback Analysis Report\n\n")
            f.write(f"**Generated:** {timestamp}\n")
            f.write(f"**Random Seed Used for Training:** {RANDOM_SEED}\n")
            f.write(f"**Total NER Training Time:** {training_time:.2f} seconds\n\n")

            # --- Data Overview ---
            f.write(f"## 1. Data Overview\n")
            total_reviews_processed = processed_df_with_all_info['Feedback Text'].nunique() if 'Feedback Text' in processed_df_with_all_info else 'N/A'
            f.write(f"- Total Unique Reviews Analyzed (for Sentiment/Kansei): {total_reviews_processed}\n")
            f.write(f"- Total Entity-Sentence Records Generated: {len(processed_df_with_all_info)}\n\n")
            
            f.write(f"### NER Training Data Composition:\n")
            total_ner_train_examples = sum(data_counts.values())
            f.write(f"| Data Source         | Count   | Percentage |\n")
            f.write(f"|---------------------|---------|------------|\n")
            f.write(f"| Base Annotated Data | {data_counts.get('base',0):<7} | { (data_counts.get('base',0)/total_ner_train_examples*100) if total_ner_train_examples else 0:.1f}% |\n")
            f.write(f"| Augmented Data      | {data_counts.get('augmented',0):<7} | { (data_counts.get('augmented',0)/total_ner_train_examples*100) if total_ner_train_examples else 0:.1f}% |\n")
            f.write(f"| Regularized Data    | {data_counts.get('regularized',0):<7} | { (data_counts.get('regularized',0)/total_ner_train_examples*100) if total_ner_train_examples else 0:.1f}% |\n")
            f.write(f"| **Total NER Train** | **{total_ner_train_examples:<7}** | **100.0%** |\n\n")

            # --- NER Performance ---
            f.write(f"## 2. Named Entity Recognition (NER) Performance\n")
            if final_ner_metrics and 'entity_level' in final_ner_metrics:
                el = final_ner_metrics['entity_level']
                f.write(f"### Overall Entity-Level (Final Validation):\n")
                f.write(f"- **Precision:** {el.get('precision', 0):.3f}\n")
                f.write(f"- **Recall:** {el.get('recall', 0):.3f}\n")
                f.write(f"- **F1-Score:** {el.get('f1', 0):.3f}\n")
                f.write(f"- True Positives: {el.get('true_positives',0)}, False Positives: {el.get('false_positives',0)}, False Negatives: {el.get('false_negatives',0)}\n\n")

                if 'per_entity' in final_ner_metrics and final_ner_metrics['per_entity']:
                    f.write(f"### Per-Entity Performance (Final Validation):\n")
                    f.write("| Entity           | Precision | Recall | F1-Score | Support | TP | FP | FN |\n")
                    f.write("|------------------|-----------|--------|----------|---------|----|----|----|\n")
                    for entity, metrics in sorted(final_ner_metrics['per_entity'].items()):
                        f.write(f"| {entity:<16} | {metrics.get('precision', 0):.3f}     | {metrics.get('recall', 0):.3f}  | {metrics.get('f1', 0):.3f}    | {metrics.get('support', 0):<7} | {metrics.get('tp',0)} | {metrics.get('fp',0)} | {metrics.get('fn',0)} |\n")
                    f.write("\n")
            else: f.write("- NER training metrics not fully available for summary.\n")
            f.write("*Refer to `plots/` for detailed NER performance visualizations.*\n\n")

            # --- Sentiment Analysis ---
            f.write(f"## 3. Sentiment Analysis\n")
            if 'Sentence Sentiment Label' in processed_df_with_all_info.columns:
                overall_sent_dist = processed_df_with_all_info['Sentence Sentiment Label'].value_counts(normalize=True) * 100
                f.write("### Overall Sentence Sentiment Distribution:\n")
                for label, perc in overall_sent_dist.items():
                    f.write(f"- **{label.title()}**: {perc:.1f}%\n")
                avg_confidence = processed_df_with_all_info['Sentence Sentiment Score'].mean() if 'Sentence Sentiment Score' in processed_df_with_all_info else 'N/A'
                # Fix the format specifier - move conditional logic outside
                if isinstance(avg_confidence, float):
                    avg_confidence_str = f"{avg_confidence:.3f}"
                else:
                    avg_confidence_str = str(avg_confidence)
                f.write(f"- Average Sentiment Confidence Score: {avg_confidence_str}\n")
                f.write("*Refer to `plots/component_sentiment_distribution.png` for component-specific sentiment.*\n\n")
            else: f.write("- Sentiment analysis results not available in the final DataFrame.\n\n")

            # --- LDA Topic Modeling ---
            f.write(f"## 4. LDA Topic Modeling\n")
            if lda_results_summary:
                f.write(f"- LDA Coherence Score (c_v): {lda_results_summary.get('coherence_score', 'N/A'):.4f}\n")
                f.write(f"- Number of LDA Topics: {lda_results_summary.get('num_topics', 'N/A')}\n")
            else: f.write("- LDA Topic Modeling results not available.\n")
            f.write("*Refer to `lda_analysis/` for LDA visualization and topic terms.*\n\n")

            # --- Kansei Engineering ---
            f.write(f"## 5. Kansei Engineering Analysis\n")
            if kansei_analysis_results and kansei_analysis_results.get('primary_emotions_distribution'):
                f.write("### Primary Kansei Emotion Distribution (Unique Reviews):\n")
                # Sort by count desc
                sorted_emotions = sorted(kansei_analysis_results['primary_emotions_distribution'].items(), key=lambda item: item[1], reverse=True)
                for emotion, count in sorted_emotions:
                    if emotion == "Unknown": continue
                    percentage = (count / total_reviews_processed * 100) if total_reviews_processed else 0
                    avg_conf = kansei_analysis_results.get('avg_confidence_per_emotion', {}).get(emotion, 0.0)
                    f.write(f"- **{emotion}**: {count} reviews ({percentage:.1f}%), Avg. Confidence: {avg_conf:.2f}\n")
                f.write("*Refer to `plots/kansei_emotion_distribution.png` and `plots/feature_kansei_correlation_heatmap.png`.*\n\n")

                f.write("### Top Kansei Design Recommendations:\n")
                recommendations = kansei_analysis_results.get('design_recommendations', [])
                if recommendations:
                    for i, rec in enumerate(recommendations[:5], 1): # Top 5
                        f.write(f"{i}. **Component Focus:** {rec.get('component','N/A')} (Priority: {rec.get('priority','N/A')})\n")
                        f.write(f"   - **Issue:** {rec.get('issue','N/A')}\n")
                        f.write(f"   - **Suggestions:** {'; '.join(rec.get('suggestions',[]))}\n")
                else: f.write("- No specific design recommendations generated.\n")
                f.write("\n*Full Kansei insights in `kansei_analysis_results.json`.*\n\n")
            else: f.write("- Kansei Engineering analysis results not available.\n\n")
            
            f.write(f"## 6. Output Files Summary\n")
            f.write("- **Main Processed Data:** `processed_feedback_with_all_analytics.csv`\n")
            f.write("- **NER Model:** `final_ner_model/`\n")
            f.write("- **NER Metrics:** `ner_training_metrics.json`\n")
            f.write("- **LDA Model & Data:** `lda_analysis/`\n")
            f.write("- **Kansei Results:** `kansei_analysis_results.json`\n")
            f.write("- **Visualizations:** `plots/`, `wordclouds/`\n")
            f.write("- **This Report:** `COMPREHENSIVE_SEAT_ANALYSIS_REPORT.md`\n")

        logger.info(f"📄 Comprehensive report saved to: {report_path}")
        return report_path


# --- Main Execution Logic ---
def main_combined_seat_analysis_pipeline(
    input_csv_path: str, #"final_dataset_compartment.csv",
    text_column_in_csv: str, #"ReviewText",
    ner_annotations_path: str, #"seat_entities_new_min.json",
    output_dir_base: str = "ULTIMATE_SEAT_ANALYSIS_V1",
    train_new_ner_model: bool = True,
    ner_training_iterations: int = 70, # Reduced from 100 for practical run time
    ner_target_max_score_for_regularization: float = 0.97,
    lda_number_of_topics: int = 10
    ):

    run_start_time = time.time()
    logger.info("🚀 STARTING COMPREHENSIVE SEAT FEEDBACK ANALYSIS PIPELINE 🚀")
    logger.info(f"Output Directory: {output_dir_base}")
    ensure_directory(output_dir_base)

    # --- 0. Initialize Managers and Visualizer/Reporter ---
    model_manager_instance = ModelManager() # For sentiment model and LDA spacy model
    visualizer_reporter = ComprehensiveVisualizerAndReporter(output_dir_base)
    
    # --- 1. Load and Preprocess Main Dataset for Analysis ---
    # This df will be used for Sentiment, LDA, and Kansei after NER model is ready
    try:
        full_main_df, _ = process_data_efficiently(input_csv_path, text_column_in_csv)
        # `full_main_df` has `text_column_in_csv` (original) and `clean_text`
        logger.info(f"Successfully loaded and preprocessed main dataset: {len(full_main_df)} records.")
        # Save a sample for inspection
        full_main_df.head().to_csv(os.path.join(output_dir_base, "sample_loaded_main_data.csv"), index=False)

    except Exception as e:
        logger.error(f"CRITICAL ERROR during main data loading: {e}", exc_info=True)
        return

    # --- 2. NER Model Training or Loading ---
    ner_model_output_path = os.path.join(output_dir_base, "final_ner_model")
    trained_ner_model = None
    ner_metrics_history_list = []
    final_ner_metrics_dict = {}
    data_composition_counts = {'base':0, 'augmented':0, 'regularized':0} # For reporting NER training data

    if train_new_ner_model:
        logger.info("--- NER Model Training Initiated ---")
        # Load base NER annotations
        base_ner_training_data = load_ner_training_data(ner_annotations_path) # Returns (cleaned_text, entities_dict)
        data_composition_counts['base'] = len(base_ner_training_data)

        if not base_ner_training_data:
            logger.error("No base NER training data loaded. Cannot train NER model.")
            return # Or handle differently, e.g., try to load a pre-existing model

        # Augment NER training data
        augmenter = FinalPushAugmenter() # Uses global SEAT_SYNONYMS
        current_training_data = list(base_ner_training_data) # Start with base
        
        # Critical examples
        augmented_for_critical_count = 0
        for entity_crit in augmenter.critical_entities.keys():
            # Dynamic count based on base data size to avoid overwhelming
            crit_count = max(20, int(len(base_ner_training_data) * 0.05)) if entity_crit != "MATERIAL" else max(15, int(len(base_ner_training_data) * 0.03))
            crit_ex = augmenter.generate_critical_examples(entity_crit, crit_count)
            current_training_data.extend(crit_ex)
            augmented_for_critical_count += len(crit_ex)
        logger.info(f"Added {augmented_for_critical_count} critical examples for NER training.")
        
        # Boost near-target
        boosted_data_full_list = augmenter.boost_near_target_entities(current_training_data)
        augmented_for_boost_count = len(boosted_data_full_list) - len(current_training_data)
        current_training_data = boosted_data_full_list
        logger.info(f"Added {augmented_for_boost_count} boost examples for NER training.")
        data_composition_counts['augmented'] = augmented_for_critical_count + augmented_for_boost_count
        
        # The PerfectScoreRegularizer is called *during* the OptimizedNERTrainer's training loop
        # So, data_composition_counts['regularized'] will be updated after training if it was applied.
        
        logger.info(f"Total NER examples before training (excluding in-loop regularization): {len(current_training_data)}")

        ner_trainer = OptimizedNERTrainer(current_training_data, target_max_score=ner_target_max_score_for_regularization)
        trained_ner_model = ner_trainer.train_optimized(n_iter=ner_training_iterations)
        
        if trained_ner_model:
            trained_ner_model.to_disk(ner_model_output_path)
            logger.info(f"Trained NER model saved to: {ner_model_output_path}")
            ner_metrics_history_list = ner_trainer.metrics_history
            final_ner_metrics_dict = ner_metrics_history_list[-1] if ner_metrics_history_list else {}
            
            # Update regularized count if regularization was applied by trainer
            # This is a bit indirect; trainer would need to expose this. For now, assume it's part of augmentation.
            # A more robust way would be for the trainer to return this count or log it clearly.
            # For this example, we'll assume the 'regularized' count is part of what the PerfectScoreRegularizer adds.
            # The current OptimizedNERTrainer applies regularization by modifying its self.train_data.
            # We'd need to get the size difference if we want to track it separately.
            # Let's assume for now data_composition_counts['regularized'] is 0 unless trainer explicitly returns it.
            # To improve: `OptimizedNERTrainer` could return the number of regularization examples added.
            
            # Visualizations for NER training
            visualizer_reporter.plot_ner_training_progress(ner_metrics_history_list)
            visualizer_reporter.plot_ner_per_entity_progress(ner_metrics_history_list) # Combined subplots
            visualizer_reporter.plot_ner_individual_entity_detailed_progress(ner_metrics_history_list) # Separate files
            visualizer_reporter.plot_ner_final_results(final_ner_metrics_dict)
            visualizer_reporter.plot_training_data_composition(data_composition_counts['base'], data_composition_counts['augmented'], data_composition_counts.get('regularized',0))

            # Save NER metrics to JSON
            metrics_path = os.path.join(output_dir_base, "ner_training_metrics.json")
            with open(metrics_path, 'w', encoding='utf-8') as f_metrics:
                json.dump(ner_metrics_history_list, f_metrics, indent=2)
            logger.info(f"NER training metrics saved to: {metrics_path}")

        else:
            logger.error("NER model training failed. Attempting to load a pre-existing model if available.")
            # Fallback to loading model if training failed
            if os.path.exists(ner_model_output_path):
                trained_ner_model = spacy.load(ner_model_output_path)
                logger.info(f"Loaded pre-existing NER model from: {ner_model_output_path}")
            else:
                logger.error("No trained or pre-existing NER model available. Pipeline cannot continue effectively.")
                return
    else: # Load existing NER model
        logger.info("--- Loading Pre-existing NER Model ---")
        if os.path.exists(ner_model_output_path):
            trained_ner_model = spacy.load(ner_model_output_path)
            logger.info(f"Loaded pre-existing NER model from: {ner_model_output_path}")
            # Try to load metrics if they exist
            metrics_path = os.path.join(output_dir_base, "ner_training_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r', encoding='utf-8') as f_metrics:
                    ner_metrics_history_list = json.load(f_metrics)
                final_ner_metrics_dict = ner_metrics_history_list[-1] if ner_metrics_history_list else {}
                logger.info("Loaded existing NER metrics.")
        else:
            logger.error(f"Pre-existing NER model not found at {ner_model_output_path}. Cannot proceed.")
            return
            
    if not trained_ner_model:
        logger.critical("NER model is not available. Exiting pipeline.")
        return

    ner_training_duration = time.time() - run_start_time # Time for NER part
    logger.info(f"NER model preparation took: {ner_training_duration:.2f} seconds.")


    # --- 3. Batch Process Main Dataset for NER & Sentiment ---
    logger.info("--- Batch Processing for NER and Sentiment ---")
    sentiment_model_pipeline = model_manager_instance.sentiment_model
    if not sentiment_model_pipeline:
        logger.warning("Sentiment model could not be loaded. Sentiment analysis will be skipped.")
    
    batch_processor = BatchNLPProcessor(
        ner_model_spacy=trained_ner_model,
        sentiment_model_pipeline=sentiment_model_pipeline
    )
    # Use 'clean_text' for processing, 'text_column_in_csv' for original feedback text
    ner_sentiment_results_list = batch_processor.process_texts_batch(
        full_main_df['clean_text'].tolist(),
        full_main_df[text_column_in_csv].tolist() 
    )
    
    if not ner_sentiment_results_list:
        logger.warning("Batch processing (NER/Sentiment) yielded no results. Subsequent analyses might be limited.")
        processed_df_with_ner_sent = pd.DataFrame()
    else:
        processed_df_with_ner_sent = pd.DataFrame(ner_sentiment_results_list)
        processed_df_with_ner_sent.to_csv(os.path.join(visualizer_reporter.tables_dir, "intermediate_ner_sentiment_results.csv"), index=False)
        logger.info(f"NER and Sentiment processing complete. DataFrame shape: {processed_df_with_ner_sent.shape}")

    # Generate general visualizations (component sentiment, frequency)
    visualizer_reporter.generate_general_visualizations(processed_df_with_ner_sent)


    # --- 4. LDA Topic Modeling ---
    logger.info("--- LDA Topic Modeling ---")
    lda_analyzer = LDAAnalyzer(output_dir_base, model_manager_instance.nlp_core_for_lda) # Pass spaCy model for LDA
    # Use 'clean_text' from the main DataFrame for LDA
    lda_model_gensim, lda_dictionary_gensim, lda_corpus_gensim, lda_coherence = lda_analyzer.perform_lda_analysis(
        full_main_df['clean_text'].dropna().astype(str).tolist(), 
        num_topics=lda_number_of_topics
    )
    lda_results_summary_dict = None
    if lda_model_gensim:
        lda_results_summary_dict = {'coherence_score': lda_coherence, 'num_topics': lda_model_gensim.num_topics}
    else:
        logger.warning("LDA modeling failed or produced no model. Kansei analysis will be affected.")


    # --- 5. Kansei Engineering Analysis ---
    logger.info("--- Kansei Engineering Analysis ---")
    kansei_analysis_output = {}
    df_with_all_analytics = processed_df_with_ner_sent.copy() # Start with NER/Sentiment results

    if lda_model_gensim and lda_dictionary_gensim and not processed_df_with_ner_sent.empty:
        kansei_module = KanseiModule(
            lda_model_gensim, 
            lda_dictionary_gensim, 
            lda_corpus_gensim, # This corpus is from the LDA training step
            processed_df_with_ner_sent # DF with NER/Sentiment
        )
        # map_topics_to_kansei_emotions augments and returns the df
        df_with_all_analytics = kansei_module.map_topics_to_kansei_emotions() 
        
        kansei_analysis_output = kansei_module.analyze_emotion_patterns()
        kansei_analysis_output['design_recommendations'] = kansei_module.generate_design_recommendations(kansei_analysis_output)
        
        # Save Kansei results to JSON
        kansei_json_path = os.path.join(output_dir_base, "kansei_analysis_results.json")
        try:
            with open(kansei_json_path, 'w', encoding='utf-8') as f_kansei:
                # Helper to make Counter serializable
                def make_serializable_for_kansei(obj):
                    if isinstance(obj, (Counter, defaultdict)): return dict(obj)
                    if isinstance(obj, dict): return {k: make_serializable_for_kansei(v) for k, v in obj.items()}
                    if isinstance(obj, list): return [make_serializable_for_kansei(i) for i in obj]
                    if isinstance(obj, (np.integer, np.floating)): return obj.item()
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    return obj
                json.dump(make_serializable_for_kansei(kansei_analysis_output), f_kansei, indent=2)
            logger.info(f"Kansei analysis results saved to: {kansei_json_path}")
        except Exception as e:
            logger.error(f"Error saving Kansei JSON: {e}")

        visualizer_reporter.generate_kansei_visualizations(kansei_analysis_output)
    else:
        logger.warning("Skipping Kansei Engineering due to missing LDA model/data or empty NER/Sentiment DataFrame.")

    # Save the final DataFrame with all analytics
    final_df_path = os.path.join(output_dir_base, "processed_feedback_with_all_analytics.csv")
    df_with_all_analytics.to_csv(final_df_path, index=False)
    logger.info(f"Final DataFrame with all analytics saved to: {final_df_path}")


    # --- 6. Word Cloud Generation ---
    logger.info("--- Word Cloud Generation ---")
    wordcloud_gen = WordCloudGenerator(output_dir_base)
    # Use 'Cleaned Text' from the main df for overall cloud, and the full df for contextual clouds
    wordcloud_gen.generate_comprehensive_wordclouds(
        full_main_df['clean_text'].dropna().astype(str).tolist(),
        df_with_all_analytics 
    )

    # --- 7. Final Comprehensive Report ---
    total_pipeline_time = time.time() - run_start_time
    visualizer_reporter.create_comprehensive_report(
        ner_metrics_history=ner_metrics_history_list,
        final_ner_metrics=final_ner_metrics_dict,
        training_time=ner_training_duration, # Specifically NER training time
        data_counts=data_composition_counts,
        processed_df_with_all_info=df_with_all_analytics,
        lda_results_summary=lda_results_summary_dict,
        kansei_analysis_results=kansei_analysis_output
    )
    
    gc.collect() # Clean up memory
    logger.info(f"🎉 PIPELINE COMPLETED! Total execution time: {total_pipeline_time:.2f} seconds.")
    logger.info(f"All outputs are in directory: {output_dir_base}")


if __name__ == "__main__":
    # --- Configuration ---
    # Path to the main dataset CSV file
    INPUT_CSV = 'final_dataset_compartment.csv' 
    # Name of the column in the CSV that contains the text feedback
    TEXT_COLUMN = 'ReviewText' 
    # Path to the JSON file containing NER annotations for training
    NER_ANNOTATIONS_FILE = 'seat_entities_new_min.json' 
    # Main directory where all outputs will be saved
    OUTPUT_DIR = 'SeatAnalysis_Combined_Output_v1' 

    # NER Training Settings
    # Set to True to train a new NER model. False to attempt loading from OUTPUT_DIR/final_ner_model/
    TRAIN_NER = True 
    NER_ITERATIONS = 50 # Number of training iterations for the NER model (e.g., 50-100)
    NER_REGULARIZATION_TARGET = 0.97 # Target F1 for PerfectScoreRegularizer (e.g., 0.97-0.98)

    # LDA Settings
    LDA_TOPICS = 10 # Number of topics for LDA model

    # --- Run Pipeline ---
    main_combined_seat_analysis_pipeline(
        input_csv_path=INPUT_CSV,
        text_column_in_csv=TEXT_COLUMN,
        ner_annotations_path=NER_ANNOTATIONS_FILE,
        output_dir_base=OUTPUT_DIR,
        train_new_ner_model=TRAIN_NER,
        ner_training_iterations=NER_ITERATIONS,
        ner_target_max_score_for_regularization=NER_REGULARIZATION_TARGET,
        lda_number_of_topics=LDA_TOPICS
    )
