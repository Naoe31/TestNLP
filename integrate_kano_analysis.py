# Add this Kano Analysis Module to your existing code

class KanoAnalysisModule:
    """Dedicated Kano Analysis Module for Customer Satisfaction Categorization"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.kano_dir = os.path.join(output_dir, "kano_analysis")
        ensure_directory(self.kano_dir)
        
        # Kano categorization thresholds based on our analysis
        self.kano_thresholds = {
            'delighter_threshold': 0.70,      # >70% positive sentiment
            'performance_min': 0.40,          # 40-70% positive sentiment
            'performance_max': 0.70,
            'dissatisfier_threshold': 0.30    # >30% negative sentiment
        }
        
        # Enhanced sentiment indicators for better categorization
        self.positive_indicators = [
            'love', 'loved', 'amazing', 'excellent', 'great', 'fantastic', 'wonderful',
            'perfect', 'outstanding', 'brilliant', 'superb', 'incredible', 'awesome',
            'comfortable', 'premium', 'luxurious', 'smooth', 'soft', 'relaxing'
        ]
        
        self.negative_indicators = [
            'hate', 'terrible', 'awful', 'horrible', 'disappointing', 'uncomfortable',
            'hard', 'stiff', 'poor', 'bad', 'cheap', 'rough', 'noisy', 'cramped',
            'tight', 'narrow', 'inadequate', 'problematic', 'annoying', 'frustrating'
        ]
        
        self.intensifiers = [
            'extremely', 'very', 'incredibly', 'absolutely', 'completely', 'totally',
            'really', 'quite', 'super', 'ultra', 'highly', 'exceptionally'
        ]

    def extract_feature_sentiment_data(self, processed_df: pd.DataFrame) -> pd.DataFrame:
        """Extract and enrich feature-sentiment data for Kano analysis"""
        logger.info("🔍 Extracting feature-sentiment data for Kano analysis...")
        
        if processed_df.empty:
            logger.warning("No processed data available for Kano analysis")
            return pd.DataFrame()
        
        # Create enhanced dataset for Kano analysis
        kano_data = []
        
        # Group by original feedback text to get complete feature context
        for feedback_text, group in processed_df.groupby('Feedback Text'):
            if pd.isna(feedback_text) or not feedback_text.strip():
                continue
                
            # Get all features mentioned in this review
            features_in_review = group['Seat Component'].unique()
            
            for _, row in group.iterrows():
                feature = row['Seat Component']
                sentence_text = str(row.get('Sentence Text', '')).lower()
                sentiment_label = row.get('Sentence Sentiment Label', 'neutral')
                sentiment_score = row.get('Sentence Sentiment Score', 0.5)
                cue_word = row.get('Cue Word', '')
                
                # Enhanced sentiment analysis
                enhanced_sentiment = self._analyze_enhanced_sentiment(
                    sentence_text, sentiment_label, sentiment_score
                )
                
                # Context analysis - how many features mentioned together
                feature_context = {
                    'features_in_review': len(features_in_review),
                    'other_features': [f for f in features_in_review if f != feature],
                    'is_isolated_mention': len(features_in_review) == 1
                }
                
                kano_data.append({
                    'feedback_id': hash(feedback_text) % 1000000,  # Create unique ID
                    'feedback_text': feedback_text,
                    'feature': feature,
                    'cue_word': cue_word,
                    'sentence_text': sentence_text,
                    'original_sentiment_label': sentiment_label,
                    'original_sentiment_score': sentiment_score,
                    'enhanced_sentiment_category': enhanced_sentiment['category'],
                    'enhanced_sentiment_intensity': enhanced_sentiment['intensity'],
                    'sentiment_confidence': enhanced_sentiment['confidence'],
                    'features_count_in_review': feature_context['features_in_review'],
                    'is_isolated_mention': feature_context['is_isolated_mention'],
                    'mention_context': 'isolated' if feature_context['is_isolated_mention'] else 'combined'
                })
        
        kano_df = pd.DataFrame(kano_data)
        
        # Save raw feature-sentiment data
        kano_df.to_csv(os.path.join(self.kano_dir, "feature_sentiment_data.csv"), index=False)
        logger.info(f"📊 Extracted {len(kano_df)} feature-sentiment records")
        
        return kano_df

    def _analyze_enhanced_sentiment(self, text: str, original_label: str, original_score: float) -> Dict:
        """Enhanced sentiment analysis with intensity and confidence"""
        text_lower = text.lower()
        
        # Count sentiment indicators
        positive_count = sum(1 for word in self.positive_indicators if word in text_lower)
        negative_count = sum(1 for word in self.negative_indicators if word in text_lower)
        intensifier_count = sum(1 for word in self.intensifiers if word in text_lower)
        
        # Determine enhanced sentiment
        if positive_count > negative_count:
            if intensifier_count > 0 or positive_count >= 2:
                category = 'very_positive'
                intensity = 'high'
            else:
                category = 'positive'
                intensity = 'medium'
        elif negative_count > positive_count:
            if intensifier_count > 0 or negative_count >= 2:
                category = 'very_negative'
                intensity = 'high'
            else:
                category = 'negative'
                intensity = 'medium'
        else:
            category = 'neutral'
            intensity = 'low'
        
        # Calculate confidence based on indicators and original score
        confidence = min(1.0, (positive_count + negative_count + intensifier_count) * 0.2 + abs(original_score - 0.5))
        
        return {
            'category': category,
            'intensity': intensity,
            'confidence': confidence,
            'positive_indicators': positive_count,
            'negative_indicators': negative_count,
            'intensifiers': intensifier_count
        }

    def perform_kano_categorization(self, kano_df: pd.DataFrame) -> Dict:
        """Perform Kano categorization based on feature mention patterns and sentiment"""
        logger.info("🎯 Performing Kano categorization...")
        
        if kano_df.empty:
            return {}
        
        # Analyze each feature
        kano_results = {}
        feature_analysis = {}
        
        for feature in kano_df['feature'].unique():
            feature_data = kano_df[kano_df['feature'] == feature].copy()
            
            if len(feature_data) == 0:
                continue
            
            # Calculate sentiment distribution
            sentiment_dist = self._calculate_sentiment_distribution(feature_data)
            
            # Determine Kano category
            kano_category = self._determine_kano_category(sentiment_dist)
            
            # Calculate additional metrics
            metrics = self._calculate_feature_metrics(feature_data, sentiment_dist)
            
            feature_analysis[feature] = {
                'total_mentions': len(feature_data),
                'sentiment_distribution': sentiment_dist,
                'kano_category': kano_category['category'],
                'kano_confidence': kano_category['confidence'],
                'kano_reasoning': kano_category['reasoning'],
                'metrics': metrics,
                'sample_quotes': self._extract_sample_quotes(feature_data)
            }
        
        # Generate Kano matrix and recommendations
        kano_matrix = self._generate_kano_matrix(feature_analysis)
        recommendations = self._generate_kano_recommendations(feature_analysis)
        
        kano_results = {
            'feature_analysis': feature_analysis,
            'kano_matrix': kano_matrix,
            'recommendations': recommendations,
            'summary_statistics': self._calculate_summary_statistics(feature_analysis)
        }
        
        # Save detailed results
        self._save_kano_results(kano_results)
        
        return kano_results

    def _calculate_sentiment_distribution(self, feature_data: pd.DataFrame) -> Dict:
        """Calculate detailed sentiment distribution for a feature"""
        total = len(feature_data)
        
        # Count by enhanced sentiment categories
        sentiment_counts = feature_data['enhanced_sentiment_category'].value_counts()
        
        # Calculate percentages
        very_positive_pct = sentiment_counts.get('very_positive', 0) / total
        positive_pct = sentiment_counts.get('positive', 0) / total
        neutral_pct = sentiment_counts.get('neutral', 0) / total
        negative_pct = sentiment_counts.get('negative', 0) / total
        very_negative_pct = sentiment_counts.get('very_negative', 0) / total
        
        # Aggregate positive and negative
        total_positive_pct = very_positive_pct + positive_pct
        total_negative_pct = negative_pct + very_negative_pct
        
        # Calculate intensity scores
        high_intensity_positive = sentiment_counts.get('very_positive', 0)
        high_intensity_negative = sentiment_counts.get('very_negative', 0)
        
        return {
            'total_mentions': total,
            'very_positive_count': sentiment_counts.get('very_positive', 0),
            'positive_count': sentiment_counts.get('positive', 0),
            'neutral_count': sentiment_counts.get('neutral', 0),
            'negative_count': sentiment_counts.get('negative', 0),
            'very_negative_count': sentiment_counts.get('very_negative', 0),
            'very_positive_pct': very_positive_pct,
            'positive_pct': positive_pct,
            'neutral_pct': neutral_pct,
            'negative_pct': negative_pct,
            'very_negative_pct': very_negative_pct,
            'total_positive_pct': total_positive_pct,
            'total_negative_pct': total_negative_pct,
            'high_intensity_positive': high_intensity_positive,
            'high_intensity_negative': high_intensity_negative,
            'avg_sentiment_confidence': feature_data['sentiment_confidence'].mean()
        }

    def _determine_kano_category(self, sentiment_dist: Dict) -> Dict:
        """Determine Kano category based on sentiment distribution"""
        total_positive = sentiment_dist['total_positive_pct']
        total_negative = sentiment_dist['total_negative_pct']
        very_positive = sentiment_dist['very_positive_pct']
        very_negative = sentiment_dist['very_negative_pct']
        total_mentions = sentiment_dist['total_mentions']
        
        # Apply Kano categorization rules
        if very_positive >= 0.5 or total_positive >= self.kano_thresholds['delighter_threshold']:
            category = "Delighter"
            confidence = min(0.95, very_positive + total_positive)
            reasoning = f"High positive sentiment ({total_positive:.1%}) with {very_positive:.1%} very positive reactions"
            
        elif total_negative >= self.kano_thresholds['dissatisfier_threshold']:
            category = "Dissatisfier"
            confidence = min(0.95, total_negative)
            reasoning = f"High negative sentiment ({total_negative:.1%}) indicating customer dissatisfaction"
            
        elif (self.kano_thresholds['performance_min'] <= total_positive <= self.kano_thresholds['performance_max']) and total_negative < 0.3:
            category = "Performance"
            confidence = 0.7
            reasoning = f"Moderate positive sentiment ({total_positive:.1%}) with balanced reactions"
            
        elif total_negative < 0.2 and total_positive < 0.4:
            category = "Must-have"
            confidence = 0.6
            reasoning = f"Low sentiment variation suggests basic expectation (pos: {total_positive:.1%}, neg: {total_negative:.1%})"
            
        else:
            # Indifferent category for unclear patterns
            category = "Indifferent"
            confidence = 0.3
            reasoning = f"Mixed or unclear sentiment pattern (pos: {total_positive:.1%}, neg: {total_negative:.1%})"
        
        # Adjust confidence based on sample size
        if total_mentions < 5:
            confidence *= 0.5
            reasoning += f" (Low confidence due to small sample: {total_mentions})"
        elif total_mentions < 10:
            confidence *= 0.7
            reasoning += f" (Moderate confidence due to sample size: {total_mentions})"
        
        return {
            'category': category,
            'confidence': confidence,
            'reasoning': reasoning
        }

    def _calculate_feature_metrics(self, feature_data: pd.DataFrame, sentiment_dist: Dict) -> Dict:
        """Calculate additional metrics for feature analysis"""
        return {
            'mention_frequency': len(feature_data),
            'avg_sentiment_score': feature_data['original_sentiment_score'].mean(),
            'sentiment_std': feature_data['original_sentiment_score'].std(),
            'isolated_mentions': len(feature_data[feature_data['is_isolated_mention'] == True]),
            'combined_mentions': len(feature_data[feature_data['is_isolated_mention'] == False]),
            'isolation_rate': len(feature_data[feature_data['is_isolated_mention'] == True]) / len(feature_data),
            'unique_feedback_count': feature_data['feedback_id'].nunique(),
            'mentions_per_feedback': len(feature_data) / feature_data['feedback_id'].nunique() if feature_data['feedback_id'].nunique() > 0 else 0
        }

    def _extract_sample_quotes(self, feature_data: pd.DataFrame, max_quotes: int = 3) -> Dict:
        """Extract representative quotes for each sentiment category"""
        quotes = {
            'very_positive': [],
            'positive': [],
            'negative': [],
            'very_negative': []
        }
        
        for category in quotes.keys():
            category_data = feature_data[feature_data['enhanced_sentiment_category'] == category]
            if not category_data.empty:
                # Select quotes with highest confidence
                top_quotes = category_data.nlargest(max_quotes, 'sentiment_confidence')
                quotes[category] = [
                    {
                        'text': row['sentence_text'],
                        'cue_word': row['cue_word'],
                        'confidence': row['sentiment_confidence']
                    }
                    for _, row in top_quotes.iterrows()
                ]
        
        return quotes

    def _generate_kano_matrix(self, feature_analysis: Dict) -> Dict:
        """Generate Kano matrix categorization"""
        matrix = {
            'Delighter': [],
            'Performance': [],
            'Must-have': [],
            'Dissatisfier': [],
            'Indifferent': []
        }
        
        for feature, analysis in feature_analysis.items():
            category = analysis['kano_category']
            matrix[category].append({
                'feature': feature,
                'confidence': analysis['kano_confidence'],
                'total_mentions': analysis['total_mentions'],
                'positive_pct': analysis['sentiment_distribution']['total_positive_pct'],
                'negative_pct': analysis['sentiment_distribution']['total_negative_pct']
            })
        
        # Sort by confidence within each category
        for category in matrix:
            matrix[category].sort(key=lambda x: x['confidence'], reverse=True)
        
        return matrix

    def _generate_kano_recommendations(self, feature_analysis: Dict) -> List[Dict]:
        """Generate actionable recommendations based on Kano analysis"""
        recommendations = []
        
        # Priority 1: Fix Dissatisfiers
        dissatisfiers = [f for f, a in feature_analysis.items() if a['kano_category'] == 'Dissatisfier']
        for feature in dissatisfiers:
            analysis = feature_analysis[feature]
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Fix Dissatisfier',
                'feature': feature,
                'action': 'Immediate improvement required',
                'rationale': f"Customer dissatisfaction ({analysis['sentiment_distribution']['total_negative_pct']:.1%} negative sentiment)",
                'expected_impact': 'Prevent customer complaints and improve basic satisfaction',
                'implementation': 'Address quality issues and design flaws'
            })
        
        # Priority 2: Enhance Delighters
        delighters = [f for f, a in feature_analysis.items() if a['kano_category'] == 'Delighter']
        for feature in delighters:
            analysis = feature_analysis[feature]
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Enhance Delighter',
                'feature': feature,
                'action': 'Maintain and promote this feature',
                'rationale': f"High customer delight ({analysis['sentiment_distribution']['total_positive_pct']:.1%} positive sentiment)",
                'expected_impact': 'Differentiate from competitors and increase customer loyalty',
                'implementation': 'Highlight in marketing and consider expanding this feature'
            })
        
        # Priority 3: Optimize Performance Features
        performance_features = [f for f, a in feature_analysis.items() if a['kano_category'] == 'Performance']
        for feature in performance_features:
            analysis = feature_analysis[feature]
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Optimize Performance',
                'feature': feature,
                'action': 'Continuously improve',
                'rationale': f"Linear satisfaction relationship - more is better",
                'expected_impact': 'Proportional increase in customer satisfaction',
                'implementation': 'Incremental improvements and benchmarking'
            })
        
        # Priority 4: Monitor Must-haves
        must_haves = [f for f, a in feature_analysis.items() if a['kano_category'] == 'Must-have']
        for feature in must_haves:
            recommendations.append({
                'priority': 'LOW',
                'category': 'Monitor Must-have',
                'feature': feature,
                'action': 'Maintain current performance',
                'rationale': 'Basic expectation - must work properly',
                'expected_impact': 'Prevent dissatisfaction',
                'implementation': 'Quality assurance and reliability testing'
            })
        
        return recommendations

    def _calculate_summary_statistics(self, feature_analysis: Dict) -> Dict:
        """Calculate summary statistics for the Kano analysis"""
        total_features = len(feature_analysis)
        category_counts = {}
        
        for analysis in feature_analysis.values():
            category = analysis['kano_category']
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            'total_features_analyzed': total_features,
            'category_distribution': category_counts,
            'category_percentages': {k: v/total_features for k, v in category_counts.items()},
            'avg_confidence': sum(a['kano_confidence'] for a in feature_analysis.values()) / total_features if total_features > 0 else 0,
            'total_mentions': sum(a['total_mentions'] for a in feature_analysis.values()),
            'features_needing_attention': len([f for f, a in feature_analysis.items() if a['kano_category'] in ['Dissatisfier', 'Indifferent']])
        }

    def _save_kano_results(self, kano_results: Dict):
        """Save Kano analysis results to files"""
        logger.info("💾 Saving Kano analysis results...")
        
        # Save complete results as JSON
        with open(os.path.join(self.kano_dir, "kano_analysis_results.json"), 'w', encoding='utf-8') as f:
            def make_serializable(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [make_serializable(i) for i in obj]
                return obj
            json.dump(make_serializable(kano_results), f, indent=2)
        
        # Save Kano matrix as CSV
        matrix_data = []
        for category, features in kano_results['kano_matrix'].items():
            for feature_data in features:
                matrix_data.append({
                    'Feature': feature_data['feature'],
                    'Kano_Category': category,
                    'Confidence': feature_data['confidence'],
                    'Total_Mentions': feature_data['total_mentions'],
                    'Positive_Percentage': feature_data['positive_pct'],
                    'Negative_Percentage': feature_data['negative_pct']
                })
        
        pd.DataFrame(matrix_data).to_csv(
            os.path.join(self.kano_dir, "kano_matrix.csv"), index=False
        )
        
        # Save recommendations as CSV
        pd.DataFrame(kano_results['recommendations']).to_csv(
            os.path.join(self.kano_dir, "kano_recommendations.csv"), index=False
        )
        
        logger.info("✅ Kano analysis results saved")

    def generate_kano_visualizations(self, kano_results: Dict):
        """Generate comprehensive Kano visualizations"""
        logger.info("📊 Generating Kano visualizations...")
        
        if not kano_results:
            logger.warning("No Kano results to visualize")
            return
        
        # 1. Kano Category Distribution
        self._plot_kano_distribution(kano_results)
        
        # 2. Feature-wise Kano Matrix
        self._plot_kano_matrix(kano_results)
        
        # 3. Sentiment vs Kano Category
        self._plot_sentiment_kano_relationship(kano_results)
        
        # 4. Priority Action Matrix
        self._plot_priority_matrix(kano_results)
        
        logger.info("✅ Kano visualizations completed")

    def _plot_kano_distribution(self, kano_results: Dict):
        """Plot Kano category distribution"""
        category_dist = kano_results['summary_statistics']['category_distribution']
        
        if not category_dist:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Create pie chart
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        wedges, texts, autotexts = plt.pie(
            category_dist.values(), 
            labels=category_dist.keys(),
            autopct='%1.1f%%',
            colors=colors,
            explode=(0.1 if 'Dissatisfier' in category_dist else 0, 0, 0, 0, 0)[:len(category_dist)]
        )
        
        plt.title('Kano Category Distribution', fontsize=16, fontweight='bold', pad=20)
        
        # Add category descriptions
        descriptions = {
            'Delighter': 'Exceed expectations',
            'Performance': 'More is better',
            'Must-have': 'Basic requirements',
            'Dissatisfier': 'Fix immediately',
            'Indifferent': 'No clear impact'
        }
        
        legend_labels = [f"{cat}\n({descriptions.get(cat, '')})" for cat in category_dist.keys()]
        plt.legend(wedges, legend_labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.kano_dir, "kano_category_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_kano_matrix(self, kano_results: Dict):
        """Plot Kano matrix with features positioned by sentiment"""
        matrix_data = []
        
        for category, features in kano_results['kano_matrix'].items():
            for feature_data in features:
                matrix_data.append({
                    'feature': feature_data['feature'],
                    'category': category,
                    'positive_pct': feature_data['positive_pct'],
                    'negative_pct': feature_data['negative_pct'],
                    'confidence': feature_data['confidence'],
                    'mentions': feature_data['total_mentions']
                })
        
        if not matrix_data:
            return
        
        df_matrix = pd.DataFrame(matrix_data)
        
        plt.figure(figsize=(14, 10))
        
        # Color map for categories
        category_colors = {
            'Delighter': '#FF6B6B',
            'Performance': '#4ECDC4', 
            'Must-have': '#45B7D1',
            'Dissatisfier': '#96CEB4',
            'Indifferent': '#FFEAA7'
        }
        
        # Scatter plot
        for category in df_matrix['category'].unique():
            cat_data = df_matrix[df_matrix['category'] == category]
            plt.scatter(
                cat_data['positive_pct'], 
                cat_data['negative_pct'],
                c=category_colors.get(category, 'gray'),
                s=cat_data['mentions'] * 10,  # Size by mention count
                alpha=0.7,
                label=category,
                edgecolors='black',
                linewidth=1
            )
            
            # Add feature labels
            for _, row in cat_data.iterrows():
                plt.annotate(
                    row['feature'].replace('_', '\n'), 
                    (row['positive_pct'], row['negative_pct']),
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=8,
                    ha='left'
                )
        
        # Add quadrant lines
        plt.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Dissatisfier threshold')
        plt.axvline(x=0.7, color='green', linestyle='--', alpha=0.5, label='Delighter threshold')
        
        plt.xlabel('Positive Sentiment Percentage', fontsize=12)
        plt.ylabel('Negative Sentiment Percentage', fontsize=12)
        plt.title('Kano Matrix: Features by Sentiment Pattern', fontsize=16, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.kano_dir, "kano_matrix_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_sentiment_kano_relationship(self, kano_results: Dict):
        """Plot relationship between sentiment scores and Kano categories"""
        # This would require the original feature data
        # For now, create a summary plot
        feature_analysis = kano_results['feature_analysis']
        
        categories = []
        avg_positive = []
        avg_negative = []
        confidence_scores = []
        
        for feature, analysis in feature_analysis.items():
            categories.append(analysis['kano_category'])
            avg_positive.append(analysis['sentiment_distribution']['total_positive_pct'])
            avg_negative.append(analysis['sentiment_distribution']['total_negative_pct'])
            confidence_scores.append(analysis['kano_confidence'])
        
        if not categories:
            return
        
        df_plot = pd.DataFrame({
            'Category': categories,
            'Positive_Pct': avg_positive,
            'Negative_Pct': avg_negative,
            'Confidence': confidence_scores
        })
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Average sentiment by category
        category_sentiment = df_plot.groupby('Category').agg({
            'Positive_Pct': 'mean',
            'Negative_Pct': 'mean'
        })
        
        x = range(len(category_sentiment.index))
        width = 0.35
        
        ax1.bar([i - width/2 for i in x], category_sentiment['Positive_Pct'], 
                width, label='Positive %', color='green', alpha=0.7)
        ax1.bar([i + width/2 for i in x], category_sentiment['Negative_Pct'], 
                width, label='Negative %', color='red', alpha=0.7)
        
        ax1.set_xlabel('Kano Category')
        ax1.set_ylabel('Average Sentiment Percentage')
        ax1.set_title('Average Sentiment by Kano Category')
        ax1.set_xticks(x)
        ax1.set_xticklabels(category_sentiment.index, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Confidence distribution
        sns.boxplot(data=df_plot, x='Category', y='Confidence', ax=ax2)
        ax2.set_title('Confidence Distribution by Kano Category')
        ax2.set_xlabel('Kano Category')
        ax2.set_ylabel('Classification Confidence')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.kano_dir, "sentiment_kano_relationship.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_priority_matrix(self, kano_results: Dict):
        """Plot priority action matrix"""
        recommendations = kano_results['recommendations']
        
        if not recommendations:
            return
        
        # Group by priority and category
        priority_counts = {}
        for rec in recommendations:
            priority = rec['priority']
            category = rec['category']
            if priority not in priority_counts:
                priority_counts[priority] = {}
            priority_counts[priority][category] = priority_counts[priority].get(category, 0) + 1
        
        # Create stacked bar chart
        priorities = list(priority_counts.keys())
        categories = set()
        for p_dict in priority_counts.values():
            categories.update(p_dict.keys())
        categories = list(categories)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bottom = [0] * len(priorities)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, category in enumerate(categories):
            values = [priority_counts[p].get(category, 0) for p in priorities]
            ax.bar(priorities, values, bottom=bottom, label=category, 
                   color=colors[i % len(colors)], alpha=0.8)
            
            # Update bottom for stacking
            bottom = [b + v for b, v in zip(bottom, values)]
        
        ax.set_xlabel('Priority Level')
        ax.set_ylabel('Number of Recommendations')
        ax.set_title('Action Priority Matrix')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, priority in enumerate(priorities):
            total = sum(priority_counts[priority].values())
            ax.text(i, total + 0.1, str(total), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.kano_dir, "priority_action_matrix.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def generate_kano_report(self, kano_results: Dict) -> str:
        """Generate comprehensive Kano analysis report"""
        logger.info("📄 Generating Kano analysis report...")
        
        report_path = os.path.join(self.kano_dir, "KANO_ANALYSIS_REPORT.md")
        
        with open(report_path, "w", encoding='utf-8') as f:
            f.write("# Kano Analysis Report\n\n")
            f.write(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## 🎯 Executive Summary\n\n")
            summary = kano_results['summary_statistics']
            f.write(f"- **Total Features Analyzed:** {summary['total_features_analyzed']}\n")
            f.write(f"- **Total Customer Mentions:** {summary['total_mentions']:,}\n")
            f.write(f"- **Average Classification Confidence:** {summary['avg_confidence']:.2f}\n")
            f.write(f"- **Features Requiring Attention:** {summary['features_needing_attention']}\n\n")
            
            # Kano Category Distribution
            f.write("## 📊 Kano Category Distribution\n\n")
            for category, count in summary['category_distribution'].items():
                percentage = summary['category_percentages'][category] * 100
                f.write(f"- **{category}:** {count} features ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Detailed Feature Analysis
            f.write("## 🔍 Detailed Feature Analysis\n\n")
            
            feature_analysis = kano_results['feature_analysis']
            
            # Group by category
            for category in ['Dissatisfier', 'Must-have', 'Performance', 'Delighter', 'Indifferent']:
                category_features = {f: a for f, a in feature_analysis.items() if a['kano_category'] == category}
                
                if category_features:
                    f.write(f"### {category} Features\n\n")
                    
                    for feature, analysis in sorted(category_features.items(), 
                                                  key=lambda x: x[1]['kano_confidence'], reverse=True):
                        f.write(f"#### {feature}\n")
                        f.write(f"- **Classification Confidence:** {analysis['kano_confidence']:.2f}\n")
                        f.write(f"- **Total Mentions:** {analysis['total_mentions']}\n")
                        f.write(f"- **Positive Sentiment:** {analysis['sentiment_distribution']['total_positive_pct']:.1%}\n")
                        f.write(f"- **Negative Sentiment:** {analysis['sentiment_distribution']['total_negative_pct']:.1%}\n")
                        f.write(f"- **Reasoning:** {analysis['kano_reasoning']}\n")
                        
                        # Add sample quotes if available
                        quotes = analysis['sample_quotes']
                        if quotes['very_positive']:
                            f.write(f"- **Example Positive Feedback:** \"{quotes['very_positive'][0]['text'][:100]}...\"\n")
                        if quotes['very_negative']:
                            f.write(f"- **Example Negative Feedback:** \"{quotes['very_negative'][0]['text'][:100]}...\"\n")
                        f.write("\n")
            
            # Recommendations
            f.write("## 🚀 Strategic Recommendations\n\n")
            
            recommendations = kano_results['recommendations']
            
            # Group by priority
            for priority in ['HIGH', 'MEDIUM', 'LOW']:
                priority_recs = [r for r in recommendations if r['priority'] == priority]
                
                if priority_recs:
                    f.write(f"### {priority} Priority Actions\n\n")
                    
                    for i, rec in enumerate(priority_recs, 1):
                        f.write(f"{i}. **{rec['feature']}** ({rec['category']})\n")
                        f.write(f"   - **Action:** {rec['action']}\n")
                        f.write(f"   - **Rationale:** {rec['rationale']}\n")
                        f.write(f"   - **Expected Impact:** {rec['expected_impact']}\n")
                        f.write(f"   - **Implementation:** {rec['implementation']}\n\n")
            
            # Generated Files
            f.write("## 📁 Generated Files\n\n")
            f.write("- `kano_analysis_results.json` - Complete analysis results\n")
            f.write("- `kano_matrix.csv` - Feature categorization matrix\n")
            f.write("- `kano_recommendations.csv` - Action recommendations\n")
            f.write("- `feature_sentiment_data.csv` - Raw feature-sentiment data\n")
            f.write("- `kano_category_distribution.png` - Category distribution chart\n")
            f.write("- `kano_matrix_plot.png` - Feature positioning matrix\n")
            f.write("- `sentiment_kano_relationship.png` - Sentiment analysis by category\n")
            f.write("- `priority_action_matrix.png` - Priority recommendations chart\n\n")
            
            # Next Steps
            f.write("## 📋 Next Steps\n\n")
            f.write("1. **Immediate Actions:** Address all Dissatisfier features\n")
            f.write("2. **Short-term:** Enhance Delighter features for competitive advantage\n")
            f.write("3. **Medium-term:** Optimize Performance features based on benchmarking\n")
            f.write("4. **Long-term:** Monitor Must-have features for quality assurance\n")
            f.write("5. **Continuous:** Re-evaluate with new customer feedback data\n\n")
        
        logger.info(f"📄 Kano report saved to {report_path}")
        return report_path

# Integration function to add to your main analysis
def integrate_kano_analysis(processed_df: pd.DataFrame, output_dir: str) -> Dict:
    """
    Integrate Kano analysis with your existing Ultimate Seat Analysis
    
    Args:
        processed_df: DataFrame from your NER and sentiment analysis
        output_dir: Output directory for results
    
    Returns:
        Dict containing Kano analysis results
    """
    logger.info("🎯 Starting Kano Analysis Integration...")
    
    # Initialize Kano module
    kano_module = KanoAnalysisModule(output_dir)
    
    # Extract feature-sentiment data
    kano_df = kano_module.extract_feature_sentiment_data(processed_df)
    
    if kano_df.empty:
        logger.warning("No data available for Kano analysis")
        return {}
    
    # Perform Kano categorization
    kano_results = kano_module.perform_kano_categorization(kano_df)
    
    if not kano_results:
        logger.warning("Kano categorization failed")
        return {}
    
    # Generate visualizations
    kano_module.generate_kano_visualizations(kano_results)
    
    # Generate report
    report_path = kano_module.generate_kano_report(kano_results)
    
    logger.info("✅ Kano analysis integration completed")
    
    # Add summary to main logger
    summary = kano_results['summary_statistics']
    logger.info(f"🎯 Kano Analysis Summary:")
    logger.info(f"   📊 Features analyzed: {summary['total_features_analyzed']}")
    logger.info(f"   🚨 Dissatisfiers: {summary['category_distribution'].get('Dissatisfier', 0)}")
    logger.info(f"   ⭐ Delighters: {summary['category_distribution'].get('Delighter', 0)}")
    logger.info(f"   📈 Performance: {summary['category_distribution'].get('Performance', 0)}")
    logger.info(f"   ✅ Must-haves: {summary['category_distribution'].get('Must-have', 0)}")
    
    return kano_results

# Modify your main function to include Kano analysis
def run_ultimate_seat_analysis_with_kano(
    csv_path: str = "final_dataset_compartment.csv",
    text_col: str = "ReviewText",
    annotations_path: str = "seat_entities_new_min.json",
    output_base_dir: str = "ultimate_seat_analysis_output",
    train_ner: bool = True,
    ner_iterations: int = 100,
    target_max_score: float = 0.98
    ):
    
    # Run your existing analysis (copy your existing function content here)
    results = run_ultimate_seat_analysis(
        csv_path, text_col, annotations_path, output_base_dir,
        train_ner, ner_iterations, target_max_score
    )
    
    if results and not results['results_df'].empty:
        # Add Kano analysis
        logger.info("\n" + "="*50)
        logger.info("🎯 ADDING KANO ANALYSIS")
        logger.info("="*50)
        
        kano_results = integrate_kano_analysis(
            results['results_df'], 
            results['output_dir']
        )
        
        # Add Kano results to main results
        results['kano_analysis'] = kano_results
        
        logger.info("🏆 ULTIMATE SEAT ANALYSIS WITH KANO COMPLETED!")
    
    return results