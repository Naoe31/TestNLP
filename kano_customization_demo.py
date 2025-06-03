"""
Kano Model Customization Demo
Shows how to create custom Kano diagrams for specific seat-related examples
"""

import matplotlib.pyplot as plt
import numpy as np

def create_seat_specific_kano_diagram():
    """Create a Kano diagram specifically for PT KAI Suite Class seats"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define the x-axis range
    x = np.linspace(-1, 1, 100)
    
    # Define Kano curves
    must_be_curve = np.where(x < 0, -1 + 0.3 * np.exp(3 * x), 0.3 * (1 - np.exp(-3 * x)))
    one_dimensional_curve = 0.8 * x
    attractive_curve = np.where(x < 0, -0.2 * (1 - np.exp(3 * x)), 1 - np.exp(-3 * x))
    
    # Plot the curves
    ax.plot(x, must_be_curve, 'r-', linewidth=4, label='Threshold (Must-be)')
    ax.plot(x, one_dimensional_curve, 'b-', linewidth=4, label='Performance (One-dimensional)')
    ax.plot(x, attractive_curve, 'g-', linewidth=4, label='Excitement (Attractive)')
    
    # Add axes
    ax.axhline(y=0, color='black', linewidth=2)
    ax.axvline(x=0, color='black', linewidth=2)
    
    # Define seat-specific attributes with positions
    threshold_attributes = [
        ("Clean seats", 0.6, 0.18),
        ("Basic cushioning", 0.4, 0.12),
        ("Seat stability", 0.7, 0.21),
        ("Adequate legroom", -0.5, -0.7)
    ]
    
    performance_attributes = [
        ("Material quality", 0.5, 0.4),
        ("Lumbar support", 0.3, 0.24),
        ("Armrest comfort", -0.3, -0.24),
        ("Headrest adjustability", 0.6, 0.48)
    ]
    
    excitement_attributes = [
        ("Massage function", 0.7, 0.85),
        ("Seat warmer", 0.5, 0.65),
        ("Premium leather", 0.6, 0.75),
        ("Memory settings", 0.4, 0.55),
        ("Footrest extension", -0.2, -0.1)
    ]
    
    # Plot threshold attributes (red)
    for attr, x_pos, y_pos in threshold_attributes:
        ax.scatter(x_pos, y_pos, s=250, c='red', alpha=0.7, edgecolors='darkred', 
                  linewidth=2, zorder=5)
        ax.annotate(attr, (x_pos, y_pos), fontsize=10, ha='center', va='bottom',
                   xytext=(0, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                            alpha=0.9, edgecolor='red'))
    
    # Plot performance attributes (blue)
    for attr, x_pos, y_pos in performance_attributes:
        ax.scatter(x_pos, y_pos, s=250, c='blue', alpha=0.7, edgecolors='darkblue', 
                  linewidth=2, zorder=5)
        ax.annotate(attr, (x_pos, y_pos), fontsize=10, ha='center', va='bottom',
                   xytext=(0, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                            alpha=0.9, edgecolor='blue'))
    
    # Plot excitement attributes (green)
    for attr, x_pos, y_pos in excitement_attributes:
        ax.scatter(x_pos, y_pos, s=250, c='green', alpha=0.7, edgecolors='darkgreen', 
                  linewidth=2, zorder=5)
        ax.annotate(attr, (x_pos, y_pos), fontsize=10, ha='center', va='bottom',
                   xytext=(0, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                            alpha=0.9, edgecolor='green'))
    
    # Add category labels with colored boxes
    ax.text(0.72, 0.92, 'EXCITEMENT ATTRIBUTES', fontsize=13, fontweight='bold', 
            color='white', bbox=dict(boxstyle='round,pad=0.6', facecolor='green', alpha=0.85),
            transform=ax.transAxes)
    
    ax.text(0.72, 0.52, 'PERFORMANCE ATTRIBUTES', fontsize=13, fontweight='bold',
            color='white', bbox=dict(boxstyle='round,pad=0.6', facecolor='blue', alpha=0.85),
            transform=ax.transAxes)
    
    ax.text(0.72, 0.12, 'THRESHOLD ATTRIBUTES', fontsize=13, fontweight='bold',
            color='white', bbox=dict(boxstyle='round,pad=0.6', facecolor='red', alpha=0.85),
            transform=ax.transAxes)
    
    # Add descriptive text
    ax.text(-0.9, -0.9, 'Basic expectations\nnot met', fontsize=11, ha='center', 
            style='italic', color='gray')
    ax.text(0.9, 0.9, 'Delighted\ncustomers', fontsize=11, ha='center', 
            style='italic', color='gray')
    
    # Labels and formatting
    ax.set_xlabel('Expectations not met ← → Expectations exceeded', 
                 fontsize=15, fontweight='bold')
    ax.set_ylabel('Customer dissatisfied ← → Customer satisfied', 
                 fontsize=15, fontweight='bold')
    ax.set_title('Kano Model Analysis - PT KAI Suite Class Seat Attributes', 
                fontsize=20, fontweight='bold', pad=25)
    
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
    
    # Add watermark/note
    ax.text(0.02, 0.02, 'Based on customer feedback analysis', 
            fontsize=9, style='italic', color='gray', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig('kano_seat_specific_demo.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("✅ Custom Kano diagram saved as 'kano_seat_specific_demo.png'")


def create_comparison_kano_diagram():
    """Create a Kano diagram comparing current vs. ideal state"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    for ax, title, alpha in [(ax1, "Current State", 0.6), (ax2, "Target State", 1.0)]:
        # Define the x-axis range
        x = np.linspace(-1, 1, 100)
        
        # Define Kano curves
        must_be_curve = np.where(x < 0, -1 + 0.3 * np.exp(3 * x), 0.3 * (1 - np.exp(-3 * x)))
        one_dimensional_curve = 0.8 * x
        attractive_curve = np.where(x < 0, -0.2 * (1 - np.exp(3 * x)), 1 - np.exp(-3 * x))
        
        # Plot the curves
        ax.plot(x, must_be_curve, 'r-', linewidth=3, alpha=alpha)
        ax.plot(x, one_dimensional_curve, 'b-', linewidth=3, alpha=alpha)
        ax.plot(x, attractive_curve, 'g-', linewidth=3, alpha=alpha)
        
        # Add axes
        ax.axhline(y=0, color='black', linewidth=1.5)
        ax.axvline(x=0, color='black', linewidth=1.5)
        
        # Current state positions (left plot)
        if ax == ax1:
            positions = {
                "Material": (-0.3, -0.24),  # Poor performance
                "Seat Warmer": (-0.8, -0.1),  # Not implemented
                "Cushion": (0.2, 0.06),  # Basic implementation
                "Massage": (-0.9, -0.05)  # Not available
            }
        else:
            # Target state positions (right plot)
            positions = {
                "Material": (0.7, 0.56),  # Improved
                "Seat Warmer": (0.6, 0.75),  # Implemented
                "Cushion": (0.8, 0.24),  # Enhanced
                "Massage": (0.7, 0.85)  # Added feature
            }
        
        # Plot components
        colors = {"Material": "blue", "Seat Warmer": "green", 
                 "Cushion": "red", "Massage": "green"}
        
        for component, (x_pos, y_pos) in positions.items():
            ax.scatter(x_pos, y_pos, s=200, c=colors[component], 
                      alpha=0.7 * alpha, edgecolors='black', linewidth=2)
            ax.annotate(component, (x_pos, y_pos), fontsize=10, ha='center',
                       xytext=(0, 10), textcoords='offset points')
        
        ax.set_xlabel('Implementation Level', fontsize=12)
        ax.set_ylabel('Customer Satisfaction', fontsize=12)
        ax.set_title(f'{title}', fontsize=16, fontweight='bold')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Kano Model: Current vs. Target State Analysis', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('kano_comparison_demo.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("✅ Comparison Kano diagram saved as 'kano_comparison_demo.png'")


if __name__ == "__main__":
    print("Creating custom Kano diagrams for PT KAI Suite Class seats...")
    print("=" * 60)
    
    # Create seat-specific diagram
    create_seat_specific_kano_diagram()
    
    # Create comparison diagram
    create_comparison_kano_diagram()
    
    print("\n" + "=" * 60)
    print("✅ All custom Kano diagrams have been generated!")
    print("\nThese visualizations can be used in your thesis to show:")
    print("1. Quality attribute categorization without specifications")
    print("2. Current vs. target state analysis")
    print("3. Priority areas for improvement") 