# Comprehensive Seat Analysis Report

**Generated on:** 2025-06-02 03:58:37

## 1. Overview
- Total Unique Reviews Analyzed: 868
- Total Entity Mentions Processed: 1,608

## 2. Named Entity Recognition (NER) Performance
### Overall Entity-Level (Final Evaluation):
- **Precision:** 0.857
- **Recall:** 0.923
- **F1-Score:** 0.889
- True Positives: 60, False Positives: 10, False Negatives: 5

### Per-Entity Performance (Final Evaluation):
| Entity           | Precision | Recall | F1-Score | Support | TP | FP | FN |
|------------------|-----------|--------|----------|---------|----|----|----|
| ARMREST          | 0.917     | 1.000  | 0.957    | 11      | 11 | 1 | 0 |
| BACKREST         | 0.667     | 1.000  | 0.800    | 2       | 2 | 1 | 0 |
| CUSHION          | 1.000     | 1.000  | 1.000    | 8       | 8 | 0 | 0 |
| FOOTREST         | 0.000     | 0.000  | 0.000    | 0       | 0 | 0 | 0 |
| HEADREST         | 1.000     | 1.000  | 1.000    | 5       | 5 | 0 | 0 |
| LUMBAR_SUPPORT   | 0.500     | 0.667  | 0.571    | 3       | 2 | 2 | 1 |
| MATERIAL         | 0.667     | 0.769  | 0.714    | 13      | 10 | 5 | 3 |
| RECLINER         | 0.947     | 1.000  | 0.973    | 18      | 18 | 1 | 0 |
| SEAT_MESSAGE     | 1.000     | 0.800  | 0.889    | 5       | 4 | 0 | 1 |
| SEAT_SIZE        | 0.000     | 0.000  | 0.000    | 0       | 0 | 0 | 0 |
| SEAT_WARMER      | 0.000     | 0.000  | 0.000    | 0       | 0 | 0 | 0 |
| TRAYTABLE        | 0.000     | 0.000  | 0.000    | 0       | 0 | 0 | 0 |

*Refer to `plots/ner_training_performance.png`, `plots/ner_per_label_performance.png`, and `tables/ner_performance_metrics.csv` for more details.*

## 3. Transformer-Based Sentiment Analysis
### Overall Sentence Sentiment Distribution:
- **Positive**: 78.1%
- **Negative**: 21.9%
- Average Sentiment Confidence Score: 0.995
*Refer to `plots/component_transformer_sentiment.png` for component-specific sentiment.*

## 4. LDA Topic Modeling and Kansei Engineering Analysis
LDA was performed to identify underlying topics, which were then mapped to Kansei emotions.
- LDA coherence score and topic count not found in lda_results.json.
*Refer to `lda_analysis/lda_visualization.html` and `lda_analysis/lda_topic_terms.csv` for LDA details.*

### Kansei Emotion Insights:
- Kansei emotion analysis results not available.

## 5. Output Files and Further Exploration
Key output files are located in the `combined_nlp_output` directory, under subfolders like `tables`, `plots`, `models`, `lda_analysis`, and `wordclouds`.
- **Detailed Data:** `processed_seat_feedback_with_kansei.csv` contains all NER, sentiment, and Kansei results per entity mention.
- **NER Model:** Trained NER model is in `models/ner_model`.
- **Kansei Insights:** `kansei_design_insights.json`.
- **Visualizations:** Various plots provide visual summaries of the findings.

