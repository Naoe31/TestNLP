#!/usr/bin/env python3
"""
Run Fixed Kano Model Analysis
This script runs the corrected Kano analysis on existing ultimate seat analysis results
"""

import os
import sys
from pathlib import Path

# Add current directory to path to import our fixed module
sys.path.insert(0, str(Path(__file__).parent))

from ultimate_seat_analysis_kano_fixed import run_kano_analysis_standalone

def main():
    """Main function to run the fixed Kano analysis"""
    
    print("=" * 70)
    print("🎯 FIXED KANO MODEL ANALYSIS FOR SEAT COMPONENTS")
    print("=" * 70)
    print("\nThis script runs Kano Model analysis on existing ultimate seat analysis results.")
    print("It provides quality insights and improvement priorities for seat components.\n")
    
    # Configuration
    base_output_dir = "ultimate_seat_analysis_output"
    processed_csv_path = os.path.join(base_output_dir, "processed_seat_feedback_with_kansei.csv")
    kano_output_dir = "kano_analysis_output"
    
    # Check if required input file exists
    if not os.path.exists(processed_csv_path):
        print(f"❌ Error: Required input file not found: {processed_csv_path}")
        print("\n💡 To fix this issue:")
        print("1. Run ultimate_seat_analysis.py first to generate the required input file")
        print("2. Make sure the file 'processed_seat_feedback_with_kansei.csv' exists in the output directory")
        print("3. Then run this script again")
        return False
    
    print(f"✅ Found input file: {processed_csv_path}")
    print(f"📁 Output will be saved to: {kano_output_dir}")
    print("\n🚀 Starting Kano Analysis...")
    
    # Run the analysis
    try:
        results = run_kano_analysis_standalone(
            processed_csv_path=processed_csv_path,
            output_dir=kano_output_dir
        )
        
        if results:
            print("\n" + "=" * 70)
            print("✅ KANO ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            
            # Display summary
            summary = results['integration_summary']
            print(f"\n📊 Analysis Summary:")
            print(f"  • Total Entities Analyzed: {summary['total_entities_analyzed']}")
            print(f"  • Critical Issues Found: {len(summary['critical_must_be_entities'])}")
            
            if summary['top_priority_entities']:
                print(f"\n🏆 Top Priority Entities:")
                for i, entity in enumerate(summary['top_priority_entities'][:3], 1):
                    print(f"  {i}. {entity['Seat_Entity']} (Priority: {entity['Priority_Score']:.3f})")
            
            if summary['critical_must_be_entities']:
                print(f"\n🚨 Critical Issues Requiring Immediate Attention:")
                for entity in summary['critical_must_be_entities']:
                    print(f"  • {entity}")
            
            print(f"\n📁 Generated Files:")
            for file_type, file_path in summary['output_files'].items():
                if os.path.exists(file_path):
                    print(f"  ✅ {file_type}: {file_path}")
                else:
                    print(f"  ❌ {file_type}: {file_path} (not created)")
            
            print(f"\n💡 Next Steps:")
            print("1. Review the Kano matrix visualization to understand entity classifications")
            print("2. Read the insights report for detailed recommendations")
            print("3. Prioritize improvements based on the analysis results")
            print("4. Focus on 'Must-be' entities with low satisfaction first")
            
            return True
            
        else:
            print("\n❌ Kano analysis failed or produced no results")
            print("Check the logs above for specific error details")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("Please check that all required dependencies are installed")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 Analysis completed successfully!")
        sys.exit(0)
    else:
        print("\n💔 Analysis failed. Please check the error messages above.")
        sys.exit(1) 