"""
Run Kano Model and IPA Analysis as QFD Alternative
This script integrates with the ultimate seat analysis outputs
"""

import os
import sys
from kano_ipa_analysis import run_kano_ipa_analysis

def main():
    # Check if ultimate seat analysis has been run
    base_output_dir = "ultimate_seat_analysis_output"
    
    required_files = [
        os.path.join(base_output_dir, "processed_seat_feedback_with_kansei.csv"),
        os.path.join(base_output_dir, "kansei_design_insights.json")
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Required file not found: {file_path}")
            print("Please run the ultimate seat analysis first.")
            sys.exit(1)
    
    print("=" * 70)
    print("KANO MODEL & IPA ANALYSIS - QFD ALTERNATIVE")
    print("=" * 70)
    print("\nThis analysis provides quality insights without requiring")
    print("existing product specifications, making it perfect for your thesis!\n")
    
    # Run the analysis
    try:
        kano_results, ipa_results, roadmap = run_kano_ipa_analysis(
            processed_csv_path=required_files[0],
            kansei_insights_path=required_files[1],
            output_dir="kano_ipa_analysis_output"
        )
        
        print("\n" + "=" * 70)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\n📊 Generated Outputs:")
        print("1. Kano Model Diagram - Shows quality attribute categories")
        print("2. IPA Matrix - Shows importance vs performance")
        print("3. Quality Improvement Roadmap - Prioritized actions")
        print("4. Comprehensive Report - Full analysis documentation")
        print("\n📁 Check results in: kano_ipa_analysis_output/")
        print("\n💡 Key Advantages for Your Thesis:")
        print("- No need for existing product specifications")
        print("- Data-driven insights from customer feedback")
        print("- Clear prioritization of improvements")
        print("- Academically recognized methodologies")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 