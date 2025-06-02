# Complete Analytics Training Report

**Generated:** 2025-06-02 03:36:30  
**Random Seed:** 42  
**Training Time:** 125.31 seconds  

## 📊 Training Data Composition

| Data Type | Count | Percentage |
|-----------|-------|------------|
| Real Data (JSON) | 135 | 12.2% |
| Critical Synthetic | 608 | 55.1% |
| Boost Synthetic | 360 | 32.6% |
| **TOTAL** | **1,103** | **100.0%** |

## 🏆 Final NER Results

**Overall F1 Score:** 0.9476  
**Entities ≥ 0.9:** 9/11  

### Per-Entity Results

| Entity | F1 Score | Precision | Recall | Support | Status |
|--------|----------|-----------|---------|---------|--------|
| ARMREST | 0.9677 | 1.0000 | 0.9375 | 16 | ✅ |
| BACKREST | 0.9333 | 0.9130 | 0.9545 | 22 | ✅ |
| HEADREST | 0.9767 | 1.0000 | 0.9545 | 22 | ✅ |
| CUSHION | 0.9767 | 1.0000 | 0.9545 | 22 | ✅ |
| MATERIAL | 0.7619 | 0.7273 | 0.8000 | 10 | ❌ |
| LUMBAR_SUPPORT | 1.0000 | 1.0000 | 1.0000 | 7 | ✅ |
| RECLINER | 0.9231 | 0.9474 | 0.9000 | 20 | ✅ |
| FOOTREST | 1.0000 | 1.0000 | 1.0000 | 16 | ✅ |
| SEAT_MESSAGE | 0.8800 | 0.9167 | 0.8462 | 26 | ⚠️ |
| SEAT_WARMER | 1.0000 | 1.0000 | 1.0000 | 21 | ✅ |
| TRAYTABLE | 0.9667 | 0.9355 | 1.0000 | 29 | ✅ |

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

