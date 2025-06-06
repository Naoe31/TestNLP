# Ultimate Seat Analysis Report

**Generated on:** 2025-06-06 03:18:55
**Random Seed:** 42

## 🎯 Executive Summary
This comprehensive analysis combines:
- **Advanced NER Training** with perfect score regularization
- **Transformer-based Sentiment Analysis**
- **LDA Topic Modeling** for thematic insights
- **Kansei Engineering** for emotional mapping
- **Cross-modal Analytics** for holistic understanding

## 📊 Dataset Overview
- **Total Unique Reviews:** 885
- **Total Entity Mentions:** 1,567
- **Entities Identified:** 11 types

## 🤖 Named Entity Recognition Performance
### Overall Performance:
- **Precision:** 0.946
- **Recall:** 0.960
- **F1-Score:** 0.953

### Per-Entity Performance:
| Entity | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|----------|
| ARMREST | 1.000 | 1.000 | 1.000 | 14 |
| BACKREST | 0.950 | 1.000 | 0.974 | 19 |
| CUSHION | 0.909 | 0.833 | 0.870 | 12 |
| FOOTREST | 1.000 | 1.000 | 1.000 | 14 |
| HEADREST | 0.938 | 1.000 | 0.968 | 15 |
| LUMBAR_SUPPORT | 1.000 | 0.875 | 0.933 | 8 |
| MATERIAL | 0.935 | 0.935 | 0.935 | 31 |
| RECLINER | 0.917 | 0.957 | 0.936 | 23 |
| SEAT_MESSAGE | 0.846 | 0.957 | 0.898 | 23 |
| SEAT_WARMER | 1.000 | 0.955 | 0.977 | 22 |
| TRAYTABLE | 1.000 | 1.000 | 1.000 | 19 |

## 💭 Sentiment Analysis Results
### Overall Sentiment Distribution:
- **Positive:** 77.0%
- **Negative:** 23.0%
- **Average Confidence:** 0.995

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

