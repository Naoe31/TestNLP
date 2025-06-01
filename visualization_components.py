"""
Visualization and analytics components for the improved NER trainer.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from collections import defaultdict
import os
from typing import List, Dict, Optional
from gensim import corpora, models
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime

plt.style.use('seaborn-v0_8')

class LDAAnalyzer:
    """Handles LDA topic modeling analysis"""
    
    def __init__(self, output_dir: str):
        self.output_dir = os.path.join(output_dir, "lda_analysis")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def perform_lda_analysis(self, texts: List[str], num_topics: int = 10):
        """Perform LDA topic modeling on training texts"""
        # Tokenize and prepare texts
        tokenized_texts = [text.lower().split() for text in texts]
        dictionary = corpora.Dictionary(tokenized_texts)
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
        
        # Train LDA model
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            update_every=1,
            passes=10
        )
        
        # Save results
        self._save_topic_analysis(lda_model, dictionary)
        coherence = models.CoherenceModel(
            model=lda_model, 
            texts=tokenized_texts, 
            dictionary=dictionary, 
            coherence='c_v'
        ).get_coherence()
        
        return lda_model, dictionary, corpus, coherence
    
    def _save_topic_analysis(self, model, dictionary):
        """Save topic analysis results"""
        with open(os.path.join(self.output_dir, "topic_analysis.md"), "w") as f:
            f.write("# LDA Topic Analysis\n\n")
            for topic_id in range(model.num_topics):
                f.write(f"## Topic {topic_id + 1}\n")
                words = model.show_topic(topic_id)
                f.write("| Word | Weight |\n|------|--------|\n")
                for word, weight in words:
                    f.write(f"| {word} | {weight:.4f} |\n")
                f.write("\n")

class WordCloudGenerator:
    """Generates various word cloud visualizations"""
    
    def __init__(self, output_dir: str):
        self.output_dir = os.path.join(output_dir, "wordclouds")
        os.makedirs(self.output_dir, exist_ok=True)
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.sia = SentimentIntensityAnalyzer()
        except Exception as e:
            print(f"Warning: Sentiment analysis unavailable - {str(e)}")
            self.sia = None
            
    def generate_comprehensive_wordclouds(
        self, 
        texts: List[str], 
        entity_texts: Dict[str, List[str]] = None,
        width: int = 800,
        height: int = 400
    ):
        """Generate comprehensive set of word clouds"""
        # Overall word cloud
        combined_text = " ".join(texts)
        self._generate_wordcloud(
            combined_text,
            os.path.join(self.output_dir, "overall_wordcloud.png"),
            width=width,
            height=height
        )
        
        # Sentiment-based word clouds
        if self.sia:
            self._generate_sentiment_wordclouds(texts, width=width, height=height)
        
        # Entity-specific word clouds
        if entity_texts:
            for entity, entity_text_list in entity_texts.items():
                if entity_text_list:
                    self._generate_wordcloud(
                        " ".join(entity_text_list),
                        os.path.join(self.output_dir, f"{entity.lower()}_wordcloud.png"),
                        width=width,
                        height=height
                    )
    
    def _generate_wordcloud(self, text: str, output_path: str, width: int = 800, height: int = 400):
        """Generate and save a word cloud"""
        try:
            wordcloud = WordCloud(
                width=width,
                height=height,
                background_color='white',
                max_words=100,
                contour_width=3,
                contour_color='steelblue'
            ).generate(text)
            
            plt.figure(figsize=(width/100, height/100))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error generating word cloud: {str(e)}")
    
    def _generate_sentiment_wordclouds(self, texts: List[str], width: int = 800, height: int = 400):
        """Generate sentiment-based word clouds"""
        positive_texts = []
        negative_texts = []
        
        for text in texts:
            scores = self.sia.polarity_scores(text)
            if scores['compound'] > 0.2:
                positive_texts.append(text)
            elif scores['compound'] < -0.2:
                negative_texts.append(text)
        
        if positive_texts:
            self._generate_wordcloud(
                " ".join(positive_texts),
                os.path.join(self.output_dir, "positive_sentiment_wordcloud.png"),
                width=width,
                height=height
            )
        
        if negative_texts:
            self._generate_wordcloud(
                " ".join(negative_texts),
                os.path.join(self.output_dir, "negative_sentiment_wordcloud.png"),
                width=width,
                height=height
            )

class ComprehensiveTrainingVisualizer:
    """Enhanced visualization combining training metrics with advanced analytics"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(os.path.join(self.plots_dir, "individual_entities"), exist_ok=True)
    
    def plot_training_progress(self, metrics_history: List[Dict], title: str = "Training Progress"):
        """Plot overall training progress"""
        epochs = range(len(metrics_history))
        overall_f1 = [m['entity_level']['f1'] for m in metrics_history]
        precision = [m['entity_level']['precision'] for m in metrics_history]
        recall = [m['entity_level']['recall'] for m in metrics_history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, overall_f1, 'b-', label='F1 Score', linewidth=2)
        plt.plot(epochs, precision, 'g--', label='Precision', alpha=0.7)
        plt.plot(epochs, recall, 'r--', label='Recall', alpha=0.7)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'training_progress.png'), dpi=300)
        plt.close()
    
    def plot_per_entity_progress(self, metrics_history: List[Dict]):
        """Plot per-entity training progress"""
        entity_f1_scores = defaultdict(list)
        epochs = range(len(metrics_history))
        
        for metrics in metrics_history:
            for entity, scores in metrics['per_entity'].items():
                entity_f1_scores[entity].append(scores['f1'])
        
        plt.figure(figsize=(15, 8))
        for entity, scores in entity_f1_scores.items():
            plt.plot(epochs, scores, label=entity, alpha=0.7)
        
        plt.title('Per-Entity F1 Scores Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'per_entity_progress.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_individual_entity_progress(self, metrics_history: List[Dict]):
        """Plot individual progress for each entity"""
        individual_plots_dir = os.path.join(self.plots_dir, "individual_entities")
        os.makedirs(individual_plots_dir, exist_ok=True)
        
        entity_metrics = defaultdict(lambda: {'f1': [], 'precision': [], 'recall': []})
        epochs = range(len(metrics_history))
        
        # Collect metrics
        for metrics in metrics_history:
            for entity, scores in metrics['per_entity'].items():
                entity_metrics[entity]['f1'].append(scores['f1'])
                entity_metrics[entity]['precision'].append(scores['precision'])
                entity_metrics[entity]['recall'].append(scores['recall'])
        
        # Create individual plots
        for entity, metrics in entity_metrics.items():
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, metrics['f1'], 'b-', label='F1 Score', linewidth=2)
            plt.plot(epochs, metrics['precision'], 'g--', label='Precision', alpha=0.7)
            plt.plot(epochs, metrics['recall'], 'r--', label='Recall', alpha=0.7)
            
            plt.title(f'{entity} Training Progress', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(individual_plots_dir, f'{entity.lower()}_progress.png'), dpi=300)
            plt.close()
    
    def plot_final_results(self, final_metrics: Dict):
        """Plot final results visualization"""
        if not final_metrics or 'per_entity' not in final_metrics:
            return
        
        # Prepare data
        entities = []
        f1_scores = []
        precisions = []
        recalls = []
        supports = []
        
        for entity, metrics in final_metrics['per_entity'].items():
            entities.append(entity)
            f1_scores.append(metrics['f1'])
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            supports.append(metrics['support'])
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # F1 Scores Bar Chart
        colors = ['green' if f1 >= 0.9 else 'orange' if f1 >= 0.8 else 'red' for f1 in f1_scores]
        bars1 = ax1.bar(entities, f1_scores, color=colors, alpha=0.7)
        ax1.axhline(y=0.9, color='red', linestyle='--', alpha=0.5)
        ax1.set_title('Final F1 Scores by Entity', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1.1)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Precision vs Recall Scatter
        scatter = ax2.scatter(recalls, precisions, c=f1_scores, s=200, cmap='RdYlGn', alpha=0.7)
        ax2.set_title('Precision vs Recall', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1.1)
        ax2.set_ylim(0, 1.1)
        plt.colorbar(scatter, ax=ax2)
        
        # Support Distribution
        ax3.bar(entities, supports, color='skyblue', alpha=0.7)
        ax3.set_title('Examples per Entity', fontsize=14, fontweight='bold')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # Summary Statistics
        summary_data = {
            'Overall F1': final_metrics['entity_level']['f1'],
            'Avg F1': np.mean(f1_scores),
            'Min F1': np.min(f1_scores),
            'Max F1': np.max(f1_scores)
        }
        ax4.bar(summary_data.keys(), summary_data.values(), color='lightcoral', alpha=0.7)
        ax4.set_title('Summary Statistics', fontsize=14, fontweight='bold')
        ax4.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'final_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_data_composition(self, base_count: int, synthetic_count: int, boost_count: int = 0):
        """Plot training data composition"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Data
        labels = ['Original Data', 'Synthetic Data', 'Boost Data'] if boost_count else ['Original Data', 'Synthetic Data']
        sizes = [base_count, synthetic_count, boost_count] if boost_count else [base_count, synthetic_count]
        colors = ['lightblue', 'lightcoral', 'lightgreen'] if boost_count else ['lightblue', 'lightcoral']
        
        # Pie chart
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Data Composition', fontsize=14, fontweight='bold')
        
        # Bar chart
        x = range(len(labels))
        ax2.bar(x, sizes, color=colors, alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.set_title('Sample Counts', fontsize=14, fontweight='bold')
        
        # Add count labels
        for i, v in enumerate(sizes):
            ax2.text(i, v, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'data_composition.png'), dpi=300, bbox_inches='tight')
        plt.close()
