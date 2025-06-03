# Kano Model Analysis for Seat Components
# Uses outputs from ultimate_seat_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import defaultdict
import os

class KanoModelAnalyzer:
    """Analyzes seat components using Kano Model based on sentiment and Kansei data"""
    
    def __init__(self, processed_df_path, kansei_insights_path, output_dir="kano_analysis_output"):
        self.df = pd.read_csv(processed_df_path)
        with open(kansei_insights_path, 'r') as f:
            self.kansei_insights = json.load(f)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Kano categories
        self.kano_categories = {
            'Must-be': 'Basic expectations - dissatisfaction if absent',
            'One-dimensional': 'Linear satisfaction - more is better',
            'Attractive': 'Delighters - unexpected features that excite',
            'Indifferent': 'No significant impact on satisfaction',
            'Reverse': 'Features that cause dissatisfaction when present'
        }
    
    def calculate_kano_metrics(self):
        """Calculate metrics for Kano categorization"""
        results = defaultdict(dict)
        
        for component in self.df['Seat Component'].unique():
            if pd.isna(component):
                continue
                
            component_df = self.df[self.df['Seat Component'] == component]
            
            # Calculate sentiment metrics
            positive_ratio = (component_df['Sentence Sentiment Label'] == 'positive').mean()
            negative_ratio = (component_df['Sentence Sentiment Label'] == 'negative').mean()
            
            # Calculate frequency metrics
            total_mentions = len(component_df)
            mention_ratio = total_mentions / len(self.df)
            
            # Get Kansei emotions for this component
            kansei_emotions = self._get_component_kansei_emotions(component)
            
            # Calculate satisfaction impact
            avg_sentiment_score = component_df['Sentence Sentiment Score'].mean()
            
            results[component] = {
                'positive_ratio': positive_ratio,
                'negative_ratio': negative_ratio,
                'mention_ratio': mention_ratio,
                'total_mentions': total_mentions,
                'avg_sentiment_score': avg_sentiment_score,
                'dominant_kansei_emotions': kansei_emotions,
                'kano_category': self._determine_kano_category(
                    positive_ratio, negative_ratio, mention_ratio, kansei_emotions
                )
            }
        
        return results
    
    def _get_component_kansei_emotions(self, component):
        """Extract Kansei emotions for a component"""
        emotions = []
        
        if 'feature_kansei_emotion_correlation' in self.kansei_insights:
            component_emotions = self.kansei_insights['feature_kansei_emotion_correlation'].get(component, {})
            # Sort by frequency and get top 3
            sorted_emotions = sorted(component_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
            emotions = [emotion for emotion, count in sorted_emotions if count > 0]
        
        return emotions
    
    def _determine_kano_category(self, pos_ratio, neg_ratio, mention_ratio, kansei_emotions):
        """Determine Kano category based on metrics"""
        
        # Must-be: High negative when absent, expected feature
        if neg_ratio > 0.4 and mention_ratio > 0.05:
            return 'Must-be'
        
        # Attractive: High positive, associated with premium/innovative emotions
        elif pos_ratio > 0.7 and any(emotion in ['Premium', 'Innovative', 'Exciting'] 
                                     for emotion in kansei_emotions):
            return 'Attractive'
        
        # One-dimensional: Balanced positive/negative, linear relationship
        elif 0.3 < pos_ratio < 0.7 and mention_ratio > 0.03:
            return 'One-dimensional'
        
        # Indifferent: Low mention rate or neutral sentiment
        elif mention_ratio < 0.02 or (0.4 < pos_ratio < 0.6 and neg_ratio < 0.2):
            return 'Indifferent'
        
        # Reverse: Might indicate over-engineering or unwanted features
        elif neg_ratio > pos_ratio and neg_ratio > 0.5:
            return 'Reverse'
        
        # Default to One-dimensional
        else:
            return 'One-dimensional'
    
    def create_kano_visualization(self, kano_results):
        """Create Kano Model visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Prepare data for plotting
        components = []
        categories = []
        satisfaction_impact = []
        dissatisfaction_impact = []
        
        for component, metrics in kano_results.items():
            components.append(component)
            categories.append(metrics['kano_category'])
            satisfaction_impact.append(metrics['positive_ratio'])
            dissatisfaction_impact.append(metrics['negative_ratio'])
        
        # Plot 1: Kano Categorization
        category_counts = pd.Series(categories).value_counts()
        colors = ['#2ecc71', '#3498db', '#f39c12', '#95a5a6', '#e74c3c']
        
        ax1.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax1.set_title('Seat Components by Kano Category', fontsize=14, fontweight='bold')
        
        # Plot 2: Satisfaction vs Dissatisfaction Impact
        color_map = {
            'Must-be': '#2ecc71',
            'One-dimensional': '#3498db', 
            'Attractive': '#f39c12',
            'Indifferent': '#95a5a6',
            'Reverse': '#e74c3c'
        }
        
        for i, (comp, cat) in enumerate(zip(components, categories)):
            ax2.scatter(dissatisfaction_impact[i], satisfaction_impact[i], 
                       s=200, c=color_map[cat], alpha=0.7, edgecolors='black')
            ax2.annotate(comp, (dissatisfaction_impact[i], satisfaction_impact[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax2.set_xlabel('Dissatisfaction Impact (Negative Ratio)', fontsize=12)
        ax2.set_ylabel('Satisfaction Impact (Positive Ratio)', fontsize=12)
        ax2.set_title('Kano Model: Satisfaction vs Dissatisfaction', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add quadrant lines
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=0.3, color='gray', linestyle='--', alpha=0.5)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[cat], label=cat) 
                          for cat in color_map.keys()]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'kano_model_visualization.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_kano_report(self, kano_results):
        """Generate detailed Kano analysis report"""
        report_path = os.path.join(self.output_dir, 'kano_analysis_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Kano Model Analysis Report\n\n")
            f.write("## Executive Summary\n\n")
            
            # Categorize components
            categorized = defaultdict(list)
            for comp, metrics in kano_results.items():
                categorized[metrics['kano_category']].append((comp, metrics))
            
            # Write category summaries
            for category in ['Must-be', 'One-dimensional', 'Attractive', 'Indifferent', 'Reverse']:
                if category in categorized:
                    f.write(f"### {category} Features ({len(categorized[category])} components)\n")
                    f.write(f"*{self.kano_categories[category]}*\n\n")
                    
                    for comp, metrics in categorized[category]:
                        f.write(f"- **{comp}**\n")
                        f.write(f"  - Mentions: {metrics['total_mentions']} ({metrics['mention_ratio']:.1%} of all feedback)\n")
                        f.write(f"  - Satisfaction: {metrics['positive_ratio']:.1%} positive\n")
                        f.write(f"  - Dissatisfaction: {metrics['negative_ratio']:.1%} negative\n")
                        if metrics['dominant_kansei_emotions']:
                            f.write(f"  - Kansei Emotions: {', '.join(metrics['dominant_kansei_emotions'])}\n")
                        f.write("\n")
            
            # Design Recommendations
            f.write("\n## Design Recommendations\n\n")
            
            f.write("### Priority 1: Fix Must-be Features\n")
            for comp, metrics in categorized.get('Must-be', []):
                f.write(f"- {comp}: Address negative feedback to meet basic expectations\n")
            
            f.write("\n### Priority 2: Optimize One-dimensional Features\n")
            for comp, metrics in categorized.get('One-dimensional', []):
                f.write(f"- {comp}: Continuous improvement will linearly increase satisfaction\n")
            
            f.write("\n### Priority 3: Invest in Attractive Features\n")
            for comp, metrics in categorized.get('Attractive', []):
                f.write(f"- {comp}: Opportunity for differentiation and delight\n")
            
            f.write("\n### Consider: Review Reverse Features\n")
            for comp, metrics in categorized.get('Reverse', []):
                f.write(f"- {comp}: May be over-engineered or causing unexpected dissatisfaction\n")
        
        # Save detailed metrics as CSV
        metrics_df = pd.DataFrame.from_dict(kano_results, orient='index')
        metrics_df.to_csv(os.path.join(self.output_dir, 'kano_metrics.csv'))
        
        print(f"Kano analysis report saved to: {report_path}")
    
    def create_priority_matrix(self, kano_results):
        """Create implementation priority matrix"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate implementation priority scores
        priority_data = []
        
        for component, metrics in kano_results.items():
            # Priority score based on category and metrics
            category = metrics['kano_category']
            
            if category == 'Must-be':
                priority_score = 10 - (metrics['positive_ratio'] * 5)  # Higher priority if low satisfaction
            elif category == 'One-dimensional':
                priority_score = 7 + (metrics['mention_ratio'] * 10)
            elif category == 'Attractive':
                priority_score = 5 + (metrics['positive_ratio'] * 3)
            elif category == 'Reverse':
                priority_score = 8  # Needs attention
            else:  # Indifferent
                priority_score = 2
            
            implementation_effort = metrics['mention_ratio'] * 10  # Proxy for complexity
            
            priority_data.append({
                'Component': component,
                'Category': category,
                'Priority Score': priority_score,
                'Implementation Effort': implementation_effort,
                'Impact': metrics['positive_ratio'] - metrics['negative_ratio']
            })
        
        priority_df = pd.DataFrame(priority_data)
        
        # Create scatter plot
        color_map = {
            'Must-be': '#2ecc71',
            'One-dimensional': '#3498db',
            'Attractive': '#f39c12', 
            'Indifferent': '#95a5a6',
            'Reverse': '#e74c3c'
        }
        
        for category in color_map:
            cat_data = priority_df[priority_df['Category'] == category]
            ax.scatter(cat_data['Implementation Effort'], cat_data['Priority Score'],
                      s=abs(cat_data['Impact'])*500, c=color_map[category],
                      alpha=0.6, edgecolors='black', label=category)
        
        # Add component labels
        for _, row in priority_df.iterrows():
            ax.annotate(row['Component'], 
                       (row['Implementation Effort'], row['Priority Score']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Implementation Effort (Mention Frequency)', fontsize=12)
        ax.set_ylabel('Priority Score', fontsize=12)
        ax.set_title('Seat Component Priority Matrix\n(Bubble size = Impact)', 
                     fontsize=14, fontweight='bold')
        ax.legend(title='Kano Category')
        ax.grid(True, alpha=0.3)
        
        # Add quadrants
        ax.axhline(y=priority_df['Priority Score'].median(), color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=priority_df['Implementation Effort'].median(), color='gray', linestyle='--', alpha=0.5)
        
        # Add quadrant labels
        ax.text(0.02, 0.98, 'Quick Wins', transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', style='italic', alpha=0.7)
        ax.text(0.98, 0.98, 'Major Projects', transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right', 
                style='italic', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'priority_matrix.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save priority data
        priority_df.to_csv(os.path.join(self.output_dir, 'priority_scores.csv'), index=False)
    
    def run_analysis(self):
        """Run complete Kano analysis"""
        print("Starting Kano Model Analysis...")
        
        # Calculate Kano metrics
        kano_results = self.calculate_kano_metrics()
        
        # Generate visualizations
        self.create_kano_visualization(kano_results)
        self.create_priority_matrix(kano_results)
        
        # Generate report
        self.generate_kano_report(kano_results)
        
        print(f"Analysis complete! Results saved to: {self.output_dir}")
        
        return kano_results


# Example usage
if __name__ == "__main__":
    # Use outputs from your ultimate_seat_analysis.py
    analyzer = KanoModelAnalyzer(
        processed_df_path="ultimate_seat_analysis_output/processed_seat_feedback_with_kansei.csv",
        kansei_insights_path="ultimate_seat_analysis_output/kansei_design_insights.json",
        output_dir="kano_analysis_output"
    )
    
    results = analyzer.run_analysis() 