# Ultimate Seat Analysis Report

**Generated on:** 2025-06-09 14:18:31
**Random Seed:** 42

## 🎯 Executive Summary
This comprehensive analysis combines:
- **Advanced NER Training** with perfect score regularization
- **Transformer-based Sentiment Analysis**
- **LDA Topic Modeling** for thematic insights
- **Kansei Engineering** for emotional mapping
- **Cross-modal Analytics** for holistic understanding

## 📊 Dataset Overview
- **Total Unique Reviews:** 890
- **Total Entity Mentions:** 1,630
- **Entities Identified:** 11 types

## 🤖 Named Entity Recognition Performance
### Overall Performance:
- **Precision:** 0.974
- **Recall:** 0.959
- **F1-Score:** 0.967

### Per-Entity Performance:
| Entity | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|----------|
| ARMREST | 0.947 | 1.000 | 0.973 | 18 |
| BACKREST | 1.000 | 0.952 | 0.976 | 21 |
| CUSHION | 1.000 | 1.000 | 1.000 | 13 |
| FOOTREST | 1.000 | 1.000 | 1.000 | 14 |
| HEADREST | 1.000 | 1.000 | 1.000 | 15 |
| LUMBAR_SUPPORT | 1.000 | 0.800 | 0.889 | 10 |
| MATERIAL | 0.923 | 1.000 | 0.960 | 24 |
| RECLINER | 0.941 | 1.000 | 0.970 | 16 |
| SEAT_MESSAGE | 0.962 | 0.862 | 0.909 | 29 |
| SEAT_WARMER | 1.000 | 0.941 | 0.970 | 17 |
| TRAYTABLE | 1.000 | 1.000 | 1.000 | 20 |

## 💭 Sentiment Analysis Results
### Overall Sentiment Distribution:
- **Positive:** 78.0%
- **Negative:** 22.0%
- **Average Confidence:** 0.996

## 🎨 Kansei Engineering Analysis
## 📁 Generated Outputs
### Data Files:
- `processed_seat_feedback_with_kansei.csv` - Complete analysis results
- `kansei_design_insights.json` - Detailed Kansei insights

### Visualizations:
- `plots/training_progress.png` - Training progress with F1 scores
- `plots/per_entity_progress.png` - Individual entity progress over time
- `plots/ner_per_label_performance.png` - Final entity-wise performance
- `plots/kansei_emotion_distribution.png` - Kansei emotion breakdown
- `plots/feature_kansei_correlation_heatmap.png` - Component-emotion correlation
- `plots/component_sentiment.png` - Sentiment distribution by component
- `plots/component_frequency.png` - Component mention frequency
- `wordclouds/overall_wordcloud.png` - Overall text word cloud
- `wordclouds/positive_sentiment_wordcloud.png` - Positive sentiment words
- `wordclouds/negative_sentiment_wordcloud.png` - Negative sentiment words
- `wordclouds/[entity]_wordcloud.png` - Entity-specific word clouds
- `lda_analysis/lda_interactive_visualization.html` - Interactive LDA topics

## 🚀 Next Steps
1. **Design Implementation:** Apply Kansei recommendations to seat design
2. **Continuous Monitoring:** Track sentiment changes over time
3. **Model Refinement:** Regular retraining with new data
4. **Cross-validation:** Validate insights with user studies

