# Complete Analytics Training Report

**Generated:** 2025-06-01 03:34:13  
**Random Seed:** 42  
**Training Time:** 261.58 seconds  

## 📊 Training Data Composition

| Data Type | Count | Percentage |
|-----------|-------|------------|
| Real Data (JSON) | 135 | 10.8% |
| Critical Synthetic | 750 | 60.2% |
| Boost Synthetic | 360 | 28.9% |
| **TOTAL** | **1,245** | **100.0%** |

## 🏆 Final NER Results

**Overall F1 Score:** 0.9549  
**Entities ≥ 0.9:** 9/11  

### Per-Entity Results

| Entity | F1 Score | Precision | Recall | Support | Status |
|--------|----------|-----------|---------|---------|--------|
| ARMREST | 0.8800 | 0.7857 | 1.0000 | 11 | ⚠️ |
| BACKREST | 0.9697 | 1.0000 | 0.9412 | 17 | ✅ |
| HEADREST | 1.0000 | 1.0000 | 1.0000 | 16 | ✅ |
| CUSHION | 1.0000 | 1.0000 | 1.0000 | 18 | ✅ |
| MATERIAL | 0.8824 | 0.8824 | 0.8824 | 17 | ⚠️ |
| LUMBAR_SUPPORT | 0.9167 | 0.9167 | 0.9167 | 12 | ✅ |
| RECLINER | 1.0000 | 1.0000 | 1.0000 | 17 | ✅ |
| FOOTREST | 1.0000 | 1.0000 | 1.0000 | 15 | ✅ |
| SEAT_MESSAGE | 0.9067 | 0.9444 | 0.8718 | 39 | ✅ |
| SEAT_WARMER | 0.9583 | 1.0000 | 0.9200 | 25 | ✅ |
| TRAYTABLE | 1.0000 | 1.0000 | 1.0000 | 25 | ✅ |

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

