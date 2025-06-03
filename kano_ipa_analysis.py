# Kano Model and IPA Analysis for Seat Quality
# Alternative to QFD for thesis without exact specifications

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import os

class KanoModelAnalysis:
    """Kano Model implementation for seat quality attributes"""
    
    def __init__(self, processed_df, kansei_insights):
        self.df = processed_df
        self.kansei_insights = kansei_insights
        
    def categorize_attributes(self):
        """Categorize seat attributes based on sentiment and frequency"""
        
        # Calculate metrics for each seat component
        component_metrics = {}
        
        for component in self.df['Seat Component'].unique():
            if pd.isna(component):
                continue
                
            component_data = self.df[self.df['Seat Component'] == component]
            
            # Calculate satisfaction metrics
            positive_ratio = (component_data['Sentence Sentiment Label'] == 'positive').mean()
            negative_ratio = (component_data['Sentence Sentiment Label'] == 'negative').mean()
            frequency = len(component_data) / len(self.df)
            
            # Get associated Kansei emotions
            kansei_emotions = []
            if 'kansei_emotion' in component_data.columns:
                kansei_emotions = component_data['kansei_emotion'].value_counts().to_dict()
            
            component_metrics[component] = {
                'positive_ratio': positive_ratio,
                'negative_ratio': negative_ratio,
                'frequency': frequency,
                'mention_count': len(component_data),
                'kansei_emotions': kansei_emotions
            }
        
        # Categorize based on Kano logic
        kano_categories = {}
        
        for component, metrics in component_metrics.items():
            # High negative when absent/poor = Must-be
            if metrics['negative_ratio'] > 0.4 and metrics['frequency'] > 0.05:
                category = 'Must-be Quality'
            # Linear relationship = One-dimensional
            elif 0.2 < metrics['positive_ratio'] < 0.7 and metrics['frequency'] > 0.03:
                category = 'One-dimensional Quality'
            # High positive, low negative = Attractive
            elif metrics['positive_ratio'] > 0.6 and metrics['negative_ratio'] < 0.2:
                category = 'Attractive Quality'
            # Low frequency, mixed sentiment = Indifferent
            elif metrics['frequency'] < 0.02:
                category = 'Indifferent Quality'
            else:
                category = 'One-dimensional Quality'  # Default
            
            kano_categories[component] = {
                'category': category,
                'metrics': metrics,
                'priority': self._calculate_priority(category, metrics)
            }
        
        return kano_categories
    
    def _calculate_priority(self, category, metrics):
        """Calculate improvement priority"""
        if category == 'Must-be Quality':
            return 1 if metrics['negative_ratio'] > 0.5 else 2
        elif category == 'One-dimensional Quality':
            return 3 if metrics['positive_ratio'] < 0.5 else 4
        elif category == 'Attractive Quality':
            return 5
        else:
            return 6
    
    def generate_kano_diagram(self, kano_categories, output_path):
        """Generate Kano Model diagram with proper curves"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Define the x-axis range
        x = np.linspace(-1, 1, 100)
        
        # Define Kano curves
        # Must-be (Threshold) curve - exponential decay
        must_be_curve = np.where(x < 0, -1 + 0.3 * np.exp(3 * x), 0.3 * (1 - np.exp(-3 * x)))
        
        # One-dimensional (Performance) curve - linear
        one_dimensional_curve = 0.8 * x
        
        # Attractive (Excitement) curve - exponential growth
        attractive_curve = np.where(x < 0, -0.2 * (1 - np.exp(3 * x)), 1 - np.exp(-3 * x))
        
        # Plot the curves
        ax.plot(x, must_be_curve, 'r-', linewidth=3, label='Must-be (Threshold)')
        ax.plot(x, one_dimensional_curve, 'b-', linewidth=3, label='One-dimensional (Performance)')
        ax.plot(x, attractive_curve, 'g-', linewidth=3, label='Attractive (Excitement)')
        
        # Add axes
        ax.axhline(y=0, color='black', linewidth=1.5)
        ax.axvline(x=0, color='black', linewidth=1.5)
        
        # Categorize components for plotting
        must_be_components = []
        one_dimensional_components = []
        attractive_components = []
        indifferent_components = []
        
        for component, data in kano_categories.items():
            category = data['category']
            metrics = data['metrics']
            
            # Calculate position based on implementation (x) and satisfaction (y)
            # X-axis: Implementation level (based on positive ratio)
            # Y-axis: Satisfaction impact (based on sentiment difference)
            
            implementation = metrics['positive_ratio'] * 2 - 1  # Scale to -1 to 1
            
            if category == 'Must-be Quality':
                must_be_components.append((component, implementation, metrics))
            elif category == 'One-dimensional Quality':
                one_dimensional_components.append((component, implementation, metrics))
            elif category == 'Attractive Quality':
                attractive_components.append((component, implementation, metrics))
            else:
                indifferent_components.append((component, implementation, metrics))
        
        # Plot components along their respective curves
        component_offset = 0.08  # Vertical offset for text
        
        # Must-be components (red)
        for i, (comp, impl, metrics) in enumerate(must_be_components):
            y_pos = np.interp(impl, x, must_be_curve)
            ax.scatter(impl, y_pos, s=200, c='red', alpha=0.7, edgecolors='darkred', linewidth=2, zorder=5)
            
            # Alternate text position to avoid overlap
            text_offset = component_offset if i % 2 == 0 else -component_offset
            ax.annotate(comp, (impl, y_pos + text_offset), fontsize=9, ha='center', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red'))
        
        # One-dimensional components (blue)
        for i, (comp, impl, metrics) in enumerate(one_dimensional_components):
            y_pos = 0.8 * impl
            ax.scatter(impl, y_pos, s=200, c='blue', alpha=0.7, edgecolors='darkblue', linewidth=2, zorder=5)
            
            text_offset = component_offset if i % 2 == 0 else -component_offset
            ax.annotate(comp, (impl, y_pos + text_offset), fontsize=9, ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='blue'))
        
        # Attractive components (green)
        for i, (comp, impl, metrics) in enumerate(attractive_components):
            y_pos = np.interp(impl, x, attractive_curve)
            ax.scatter(impl, y_pos, s=200, c='green', alpha=0.7, edgecolors='darkgreen', linewidth=2, zorder=5)
            
            text_offset = component_offset if i % 2 == 0 else -component_offset
            ax.annotate(comp, (impl, y_pos + text_offset), fontsize=9, ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='green'))
        
        # Indifferent components (gray) - plot near the x-axis
        for i, (comp, impl, metrics) in enumerate(indifferent_components):
            y_pos = 0.1 * np.random.uniform(-1, 1)  # Small random y position
            ax.scatter(impl, y_pos, s=150, c='gray', alpha=0.5, edgecolors='darkgray', linewidth=2, zorder=5)
            ax.annotate(comp, (impl, y_pos + 0.05), fontsize=8, ha='center', color='gray')
        
        # Add category labels with colored boxes
        ax.text(0.7, 0.85, 'ATTRACTIVE ATTRIBUTES', fontsize=12, fontweight='bold', 
                color='white', bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.8),
                transform=ax.transAxes)
        
        ax.text(0.7, 0.5, 'PERFORMANCE ATTRIBUTES', fontsize=12, fontweight='bold',
                color='white', bbox=dict(boxstyle='round,pad=0.5', facecolor='blue', alpha=0.8),
                transform=ax.transAxes)
        
        ax.text(0.7, 0.15, 'THRESHOLD ATTRIBUTES', fontsize=12, fontweight='bold',
                color='white', bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.8),
                transform=ax.transAxes)
        
        # Labels and formatting
        ax.set_xlabel('Expectations\nnot met ← → Expectations\nexceeded', fontsize=14, fontweight='bold')
        ax.set_ylabel('Customer dissatisfied ← → Customer satisfied', fontsize=14, fontweight='bold')
        ax.set_title('Kano Model Analysis - PT KAI Suite Class Seat Components', fontsize=18, fontweight='bold', pad=20)
        
        # Set axis limits
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add arrows to axes
        ax.annotate('', xy=(1.15, 0), xytext=(-1.15, 0),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        ax.annotate('', xy=(0, 1.15), xytext=(0, -1.15),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        
        # Remove default spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Add legend
        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Also create a simplified version with just the curves
        self._generate_kano_curves_only(output_path.replace('.png', '_curves.png'))

    def _generate_kano_curves_only(self, output_path):
        """Generate a clean Kano curves diagram without data points"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define the x-axis range
        x = np.linspace(-1, 1, 100)
        
        # Define Kano curves
        must_be_curve = np.where(x < 0, -1 + 0.3 * np.exp(3 * x), 0.3 * (1 - np.exp(-3 * x)))
        one_dimensional_curve = 0.8 * x
        attractive_curve = np.where(x < 0, -0.2 * (1 - np.exp(3 * x)), 1 - np.exp(-3 * x))
        
        # Plot the curves with thicker lines
        ax.plot(x, must_be_curve, 'r-', linewidth=4, label='Threshold Attributes')
        ax.plot(x, one_dimensional_curve, 'b-', linewidth=4, label='Performance Attributes')
        ax.plot(x, attractive_curve, 'g-', linewidth=4, label='Excitement Attributes')
        
        # Add axes
        ax.axhline(y=0, color='black', linewidth=2)
        ax.axvline(x=0, color='black', linewidth=2)
        
        # Add example labels (like in the airline example)
        # Threshold examples
        ax.text(-0.7, -0.5, 'Basic comfort\nnot provided', fontsize=10, ha='center', color='darkred')
        ax.text(0.7, 0.2, 'Basic comfort\nprovided', fontsize=10, ha='center', color='darkred')
        
        # Performance examples
        ax.text(-0.5, -0.4, 'Poor quality\nmaterials', fontsize=10, ha='center', color='darkblue')
        ax.text(0.5, 0.4, 'Premium quality\nmaterials', fontsize=10, ha='center', color='darkblue')
        
        # Excitement examples
        ax.text(-0.3, -0.1, 'No massage\nfunction', fontsize=10, ha='center', color='darkgreen')
        ax.text(0.6, 0.8, 'Advanced massage\nsystem', fontsize=10, ha='center', color='darkgreen')
        
        # Labels and formatting
        ax.set_xlabel('Expectations not met ← → Expectations exceeded', fontsize=14, fontweight='bold')
        ax.set_ylabel('Customer dissatisfied ← → Customer satisfied', fontsize=14, fontweight='bold')
        ax.set_title('Kano Model - Quality Attribute Categories', fontsize=18, fontweight='bold', pad=20)
        
        # Set axis limits
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add arrows to axes
        ax.annotate('', xy=(1.15, 0), xytext=(-1.15, 0),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        ax.annotate('', xy=(0, 1.15), xytext=(0, -1.15),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # Remove default spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Add colored boxes for categories
        ax.text(0.02, 0.98, 'EXCITEMENT ATTRIBUTES', fontsize=12, fontweight='bold', 
                color='white', bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.8),
                transform=ax.transAxes, va='top')
        
        ax.text(0.02, 0.88, 'PERFORMANCE ATTRIBUTES', fontsize=12, fontweight='bold',
                color='white', bbox=dict(boxstyle='round,pad=0.5', facecolor='blue', alpha=0.8),
                transform=ax.transAxes, va='top')
        
        ax.text(0.02, 0.78, 'THRESHOLD ATTRIBUTES', fontsize=12, fontweight='bold',
                color='white', bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.8),
                transform=ax.transAxes, va='top')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()


class ImportancePerformanceAnalysis:
    """IPA implementation for seat quality attributes"""
    
    def __init__(self, processed_df):
        self.df = processed_df
    
    def calculate_ipa_metrics(self):
        """Calculate importance and performance for each component"""
        
        ipa_data = {}
        total_mentions = len(self.df)
        
        for component in self.df['Seat Component'].unique():
            if pd.isna(component):
                continue
            
            component_data = self.df[self.df['Seat Component'] == component]
            
            # Importance = frequency of mentions (normalized)
            importance = len(component_data) / total_mentions
            
            # Performance = sentiment score
            sentiment_scores = {
                'positive': 1.0,
                'neutral': 0.5,
                'negative': 0.0
            }
            
            performance_scores = component_data['Sentence Sentiment Label'].map(
                lambda x: sentiment_scores.get(x, 0.5)
            )
            performance = performance_scores.mean()
            
            # Calculate confidence
            confidence = component_data['Sentence Sentiment Score'].mean()
            
            ipa_data[component] = {
                'importance': importance,
                'performance': performance,
                'confidence': confidence,
                'count': len(component_data)
            }
        
        return ipa_data
    
    def generate_ipa_matrix(self, ipa_data, output_path):
        """Generate IPA matrix plot"""
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Extract data for plotting
        components = list(ipa_data.keys())
        importance = [ipa_data[c]['importance'] for c in components]
        performance = [ipa_data[c]['performance'] for c in components]
        counts = [ipa_data[c]['count'] for c in components]
        
        # Calculate mean lines
        mean_importance = np.mean(importance)
        mean_performance = np.mean(performance)
        
        # Create scatter plot
        scatter = ax.scatter(importance, performance, s=[c*5 for c in counts], 
                           alpha=0.6, c=performance, cmap='RdYlGn', 
                           edgecolors='black', linewidth=1)
        
        # Add component labels
        for i, component in enumerate(components):
            ax.annotate(component, (importance[i], performance[i]), 
                       fontsize=9, ha='center', va='bottom')
        
        # Add quadrant lines
        ax.axvline(mean_importance, color='gray', linestyle='--', alpha=0.7)
        ax.axhline(mean_performance, color='gray', linestyle='--', alpha=0.7)
        
        # Add quadrant labels
        ax.text(0.95, 0.95, 'Keep Up The\nGood Work', transform=ax.transAxes, 
                fontsize=12, ha='right', va='top', bbox=dict(boxstyle='round', 
                facecolor='lightgreen', alpha=0.5))
        
        ax.text(0.05, 0.95, 'Possible\nOverkill', transform=ax.transAxes, 
                fontsize=12, ha='left', va='top', bbox=dict(boxstyle='round', 
                facecolor='lightyellow', alpha=0.5))
        
        ax.text(0.05, 0.05, 'Low\nPriority', transform=ax.transAxes, 
                fontsize=12, ha='left', va='bottom', bbox=dict(boxstyle='round', 
                facecolor='lightgray', alpha=0.5))
        
        ax.text(0.95, 0.05, 'Concentrate\nHere', transform=ax.transAxes, 
                fontsize=12, ha='right', va='bottom', bbox=dict(boxstyle='round', 
                facecolor='lightcoral', alpha=0.5))
        
        ax.set_xlabel('Importance (Frequency of Mentions) →', fontsize=12)
        ax.set_ylabel('Performance (Sentiment Score) →', fontsize=12)
        ax.set_title('Importance-Performance Analysis - Seat Components', 
                    fontsize=16, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Performance Score', fontsize=10)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_action_matrix(self, ipa_data):
        """Generate actionable recommendations based on IPA"""
        
        mean_importance = np.mean([d['importance'] for d in ipa_data.values()])
        mean_performance = np.mean([d['performance'] for d in ipa_data.values()])
        
        recommendations = {
            'concentrate_here': [],
            'keep_up_good_work': [],
            'low_priority': [],
            'possible_overkill': []
        }
        
        for component, data in ipa_data.items():
            if data['importance'] > mean_importance:
                if data['performance'] > mean_performance:
                    recommendations['keep_up_good_work'].append({
                        'component': component,
                        'action': 'Maintain current quality standards',
                        'importance': data['importance'],
                        'performance': data['performance']
                    })
                else:
                    recommendations['concentrate_here'].append({
                        'component': component,
                        'action': 'Urgent improvement needed',
                        'importance': data['importance'],
                        'performance': data['performance']
                    })
            else:
                if data['performance'] > mean_performance:
                    recommendations['possible_overkill'].append({
                        'component': component,
                        'action': 'Consider resource reallocation',
                        'importance': data['importance'],
                        'performance': data['performance']
                    })
                else:
                    recommendations['low_priority'].append({
                        'component': component,
                        'action': 'Monitor but no immediate action',
                        'importance': data['importance'],
                        'performance': data['performance']
                    })
        
        return recommendations


def generate_quality_improvement_roadmap(kano_results, ipa_results, output_path):
    """Generate integrated quality improvement roadmap"""
    
    roadmap = {
        'immediate_actions': [],
        'short_term': [],
        'long_term': [],
        'monitoring': []
    }
    
    # Combine insights from both analyses
    for component in kano_results.keys():
        kano_data = kano_results[component]
        ipa_data = ipa_results.get(component, {})
        
        priority_score = kano_data['priority']
        
        # Immediate actions (Priority 1-2 or Concentrate Here)
        if priority_score <= 2 or (ipa_data and ipa_data['importance'] > 0.05 and ipa_data['performance'] < 0.5):
            roadmap['immediate_actions'].append({
                'component': component,
                'kano_category': kano_data['category'],
                'current_performance': ipa_data.get('performance', 'N/A'),
                'action': 'Immediate improvement required to meet basic expectations'
            })
        
        # Short-term improvements (Priority 3-4)
        elif priority_score <= 4:
            roadmap['short_term'].append({
                'component': component,
                'kano_category': kano_data['category'],
                'current_performance': ipa_data.get('performance', 'N/A'),
                'action': 'Enhance to improve overall satisfaction'
            })
        
        # Long-term innovations (Priority 5)
        elif priority_score == 5:
            roadmap['long_term'].append({
                'component': component,
                'kano_category': kano_data['category'],
                'current_performance': ipa_data.get('performance', 'N/A'),
                'action': 'Innovation opportunity for competitive advantage'
            })
        
        # Monitoring only
        else:
            roadmap['monitoring'].append({
                'component': component,
                'kano_category': kano_data['category'],
                'current_performance': ipa_data.get('performance', 'N/A'),
                'action': 'Monitor customer feedback'
            })
    
    # Save roadmap
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(roadmap, f, indent=2)
    
    return roadmap


def run_kano_ipa_analysis(
    processed_csv_path: str,
    kansei_insights_path: str,
    output_dir: str
):
    """Main function to run Kano and IPA analyses"""
    
    print("Starting Kano Model and IPA Analysis...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    processed_df = pd.read_csv(processed_csv_path)
    with open(kansei_insights_path, 'r') as f:
        kansei_insights = json.load(f)
    
    # Run Kano Model Analysis
    print("Running Kano Model Analysis...")
    kano_analyzer = KanoModelAnalysis(processed_df, kansei_insights)
    kano_results = kano_analyzer.categorize_attributes()
    kano_analyzer.generate_kano_diagram(
        kano_results, 
        os.path.join(output_dir, 'kano_model_diagram.png')
    )
    
    # Save Kano results
    kano_output = {}
    for component, data in kano_results.items():
        kano_output[component] = {
            'category': data['category'],
            'priority': data['priority'],
            'positive_ratio': data['metrics']['positive_ratio'],
            'negative_ratio': data['metrics']['negative_ratio'],
            'frequency': data['metrics']['frequency']
        }
    
    with open(os.path.join(output_dir, 'kano_analysis_results.json'), 'w') as f:
        json.dump(kano_output, f, indent=2)
    
    # Run IPA Analysis
    print("Running Importance-Performance Analysis...")
    ipa_analyzer = ImportancePerformanceAnalysis(processed_df)
    ipa_results = ipa_analyzer.calculate_ipa_metrics()
    ipa_analyzer.generate_ipa_matrix(
        ipa_results,
        os.path.join(output_dir, 'ipa_matrix.png')
    )
    
    # Generate recommendations
    ipa_recommendations = ipa_analyzer.generate_action_matrix(ipa_results)
    
    # Save IPA results
    with open(os.path.join(output_dir, 'ipa_analysis_results.json'), 'w') as f:
        json.dump({
            'metrics': ipa_results,
            'recommendations': ipa_recommendations
        }, f, indent=2)
    
    # Generate integrated roadmap
    print("Generating Quality Improvement Roadmap...")
    roadmap = generate_quality_improvement_roadmap(
        kano_results,
        ipa_results,
        os.path.join(output_dir, 'quality_improvement_roadmap.json')
    )
    
    # Create summary report
    create_kano_ipa_report(
        kano_results,
        ipa_results,
        ipa_recommendations,
        roadmap,
        os.path.join(output_dir, 'kano_ipa_analysis_report.md')
    )
    
    print(f"Analysis complete! Results saved to {output_dir}")
    
    return kano_results, ipa_results, roadmap


def create_kano_ipa_report(kano_results, ipa_results, ipa_recommendations, roadmap, output_path):
    """Create comprehensive report for Kano and IPA analyses"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Kano Model and IPA Analysis Report\n\n")
        f.write("## Executive Summary\n\n")
        f.write("This report presents quality analysis using Kano Model and Importance-Performance Analysis (IPA) ")
        f.write("as alternatives to traditional QFD, suitable for scenarios without existing product specifications.\n\n")
        
        # Kano Model Results
        f.write("## Kano Model Analysis\n\n")
        f.write("### Component Categories:\n\n")
        
        categories_count = defaultdict(list)
        for component, data in kano_results.items():
            categories_count[data['category']].append(component)
        
        for category, components in categories_count.items():
            f.write(f"**{category}:**\n")
            for comp in components:
                metrics = kano_results[comp]['metrics']
                f.write(f"- {comp} (Positive: {metrics['positive_ratio']:.1%}, ")
                f.write(f"Negative: {metrics['negative_ratio']:.1%})\n")
            f.write("\n")
        
        # IPA Results
        f.write("## Importance-Performance Analysis\n\n")
        
        for quadrant, items in ipa_recommendations.items():
            if items:
                f.write(f"### {quadrant.replace('_', ' ').title()}:\n")
                for item in items:
                    f.write(f"- **{item['component']}**: {item['action']}\n")
                    f.write(f"  - Importance: {item['importance']:.3f}\n")
                    f.write(f"  - Performance: {item['performance']:.3f}\n")
                f.write("\n")
        
        # Quality Improvement Roadmap
        f.write("## Quality Improvement Roadmap\n\n")
        
        if roadmap['immediate_actions']:
            f.write("### 🚨 Immediate Actions Required:\n")
            for action in roadmap['immediate_actions']:
                f.write(f"1. **{action['component']}** ({action['kano_category']})\n")
                f.write(f"   - Current Performance: {action['current_performance']}\n")
                f.write(f"   - Action: {action['action']}\n\n")
        
        if roadmap['short_term']:
            f.write("### 📅 Short-term Improvements (3-6 months):\n")
            for action in roadmap['short_term']:
                f.write(f"- **{action['component']}**: {action['action']}\n")
        
        if roadmap['long_term']:
            f.write("\n### 🎯 Long-term Innovation Opportunities:\n")
            for action in roadmap['long_term']:
                f.write(f"- **{action['component']}**: {action['action']}\n")
        
        f.write("\n## Methodology Note\n\n")
        f.write("This analysis provides actionable insights without requiring existing product specifications, ")
        f.write("making it ideal for new product development or competitive analysis scenarios.\n")


if __name__ == "__main__":
    # Use the outputs from your ultimate seat analysis
    run_kano_ipa_analysis(
        processed_csv_path="ultimate_seat_analysis_output/processed_seat_feedback_with_kansei.csv",
        kansei_insights_path="ultimate_seat_analysis_output/kansei_design_insights.json",
        output_dir="kano_ipa_analysis_output"
    ) 