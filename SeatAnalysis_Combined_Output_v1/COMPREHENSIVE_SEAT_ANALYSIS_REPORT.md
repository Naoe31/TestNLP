# Comprehensive Seat Feedback Analysis Report

**Generated:** 2025-06-02 12:47:53
**Random Seed Used for Training:** 42
**Total NER Training Time:** 141.72 seconds

## 1. Data Overview
- Total Unique Reviews Analyzed (for Sentiment/Kansei): 887
- Total Entity-Sentence Records Generated: 1706

### NER Training Data Composition:
| Data Source         | Count   | Percentage |
|---------------------|---------|------------|
| Base Annotated Data | 135     | 22.9% |
| Augmented Data      | 455     | 77.1% |
| Regularized Data    | 0       | 0.0% |
| **Total NER Train** | **590    ** | **100.0%** |

## 2. Named Entity Recognition (NER) Performance
### Overall Entity-Level (Final Validation):
- **Precision:** 0.923
- **Recall:** 0.907
- **F1-Score:** 0.915
- True Positives: 156, False Positives: 13, False Negatives: 16

### Per-Entity Performance (Final Validation):
| Entity           | Precision | Recall | F1-Score | Support | TP | FP | FN |
|------------------|-----------|--------|----------|---------|----|----|----|
| ARMREST          | 1.000     | 1.000  | 1.000    | 25      | 25 | 0 | 0 |
| BACKREST         | 1.000     | 1.000  | 1.000    | 9       | 9 | 0 | 0 |
| CUSHION          | 0.889     | 0.941  | 0.914    | 17      | 16 | 2 | 1 |
| FOOTREST         | 1.000     | 0.875  | 0.933    | 16      | 14 | 0 | 2 |
| HEADREST         | 1.000     | 1.000  | 1.000    | 19      | 19 | 0 | 0 |
| LUMBAR_SUPPORT   | 1.000     | 0.923  | 0.960    | 13      | 12 | 0 | 1 |
| MATERIAL         | 0.545     | 0.667  | 0.600    | 9       | 6 | 5 | 3 |
| RECLINER         | 0.939     | 0.969  | 0.954    | 32      | 31 | 2 | 1 |
| SEAT_MESSAGE     | 0.714     | 0.833  | 0.769    | 12      | 10 | 4 | 2 |
| SEAT_SIZE        | 0.000     | 0.000  | 0.000    | 0       | 0 | 0 | 0 |
| SEAT_WARMER      | 1.000     | 0.636  | 0.778    | 11      | 7 | 0 | 4 |
| TRAYTABLE        | 1.000     | 0.778  | 0.875    | 9       | 7 | 0 | 2 |

*Refer to `plots/` for detailed NER performance visualizations.*

## 3. Sentiment Analysis
### Overall Sentence Sentiment Distribution:
- **Positive**: 78.1%
- **Negative**: 21.9%
- Average Sentiment Confidence Score: 0.996
*Refer to `plots/component_sentiment_distribution.png` for component-specific sentiment.*

## 4. LDA Topic Modeling
- LDA Topic Modeling results not available.
*Refer to `lda_analysis/` for LDA visualization and topic terms.*

## 5. Kansei Engineering Analysis
- Kansei Engineering analysis results not available.

## 6. Output Files Summary
- **Main Processed Data:** `processed_feedback_with_all_analytics.csv`
- **NER Model:** `final_ner_model/`
- **NER Metrics:** `ner_training_metrics.json`
- **LDA Model & Data:** `lda_analysis/`
- **Kansei Results:** `kansei_analysis_results.json`
- **Visualizations:** `plots/`, `wordclouds/`
- **This Report:** `COMPREHENSIVE_SEAT_ANALYSIS_REPORT.md`
