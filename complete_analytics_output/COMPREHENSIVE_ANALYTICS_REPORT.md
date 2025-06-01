# Complete Analytics Training Report

**Generated:** 2025-06-01 04:27:52  
**Random Seed:** 42  
**Training Time:** 154.12 seconds  

## 📊 Training Data Composition

| Data Type | Count | Percentage |
|-----------|-------|------------|
| Real Data (JSON) | 135 | 12.2% |
| Critical Synthetic | 608 | 55.1% |
| Boost Synthetic | 360 | 32.6% |
| **TOTAL** | **1,103** | **100.0%** |

## 🏆 Final NER Results

**Overall F1 Score:** 0.9506  
**Entities ≥ 0.9:** 9/11  

### Per-Entity Results

| Entity | F1 Score | Precision | Recall | Support | Status |
|--------|----------|-----------|---------|---------|--------|
| ARMREST | 1.0000 | 1.0000 | 1.0000 | 15 | ✅ |
| BACKREST | 0.9500 | 0.9500 | 0.9500 | 20 | ✅ |
| HEADREST | 1.0000 | 1.0000 | 1.0000 | 18 | ✅ |
| CUSHION | 1.0000 | 1.0000 | 1.0000 | 18 | ✅ |
| MATERIAL | 0.5455 | 0.7500 | 0.4286 | 7 | ❌ |
| LUMBAR_SUPPORT | 0.7143 | 0.7143 | 0.7143 | 7 | ❌ |
| RECLINER | 0.9714 | 0.9444 | 1.0000 | 17 | ✅ |
| FOOTREST | 1.0000 | 1.0000 | 1.0000 | 19 | ✅ |
| SEAT_MESSAGE | 0.9231 | 0.9600 | 0.8889 | 27 | ✅ |
| SEAT_WARMER | 0.9474 | 1.0000 | 0.9000 | 20 | ✅ |
| TRAYTABLE | 0.9818 | 0.9643 | 1.0000 | 27 | ✅ |

## 📈 Generated Visualizations

### Training Progress
- **Overall Progress:** `plots/training_progress.png`
- **Per-Entity Progress (Combined):** `plots/per_entity_progress.png`
- **Individual Entity Progress:** `plots/individual_entities/[entity]_progress.png`
- **Final Results:** `plots/final_results.png`
- **Data Composition:** `plots/data_composition.png`

### Advanced Analytics
- **Overall Word Cloud:** `wordclouds/overall_wordcloud.png`
- **Sentiment Word Clouds:** `wordclouds/positive_sentiment_wordcloud.png`, `wordclouds/negative_sentiment_wordcloud.png`
- **Entity Word Clouds:** `wordclouds/[entity]_wordcloud.png`

## 📁 Output Structure

```
complete_analytics_output/
├── plots/                  # Training visualizations
│   ├── individual_entities/ # Individual entity progress plots
├── lda_analysis/           # LDA topic modeling results
├── wordclouds/             # Word cloud visualizations
├── analytics/              # Additional analytics
└── final_push_model_complete/  # Trained NER model
```

