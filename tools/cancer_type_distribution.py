import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from collections import Counter
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore')

def safe_literal_eval(x):
    """Safely evaluate string representations of lists/arrays"""
    if pd.isna(x) or x is None:
        return None
    if isinstance(x, str):
        if x.strip() in ['', '[]', 'nan', 'None']:
            return []
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return None
    return x

def is_non_empty_caliper_boxes(x):
    """Check if caliper_boxes contains actual data (not empty list or None)"""
    if pd.isna(x) or x is None:
        return False
    if isinstance(x, str):
        if x.strip() in ['', '[]', 'nan', 'None']:
            return False
        try:
            evaluated = ast.literal_eval(x)
            return isinstance(evaluated, list) and len(evaluated) > 0
        except (ValueError, SyntaxError):
            return False
    if isinstance(x, list):
        return len(x) > 0
    return False

def analyze_cancer_distributions(lesion_csv_path, image_csv_path):
    """
    Analyze and compare cancer type distributions between raw data and caliper detections
    """
    
    print("Loading CSV files...")
    
    # Load the CSV files
    try:
        lesion_df = pd.read_csv(lesion_csv_path)
        image_df = pd.read_csv(image_csv_path)
        print(f"✓ Loaded LesionData: {len(lesion_df)} rows")
        print(f"✓ Loaded ImageData: {len(image_df)} rows")
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return
    
    # Print column info for debugging
    print(f"\nLesionData columns: {list(lesion_df.columns)}")
    print(f"ImageData columns: {list(image_df.columns)}")
    
    # Check for the merge key columns
    lesion_acc_col = 'ACCESSION_NUMBER'
    image_acc_col = 'Accession_Number'

    print(f"\nMerging datasets on {lesion_acc_col} / {image_acc_col}...")
    
    # Merge the datasets
    merged_df = pd.merge(
        lesion_df, 
        image_df, 
        left_on=lesion_acc_col, 
        right_on=image_acc_col, 
        how='inner'
    )
    
    print(f"✓ Merged dataset: {len(merged_df)} rows")
    
    # Remove rows where cancer_type is null
    merged_df = merged_df.dropna(subset=['cancer_type'])
    print(f"✓ After removing null cancer_types: {len(merged_df)} rows")
    
    return merged_df

def proportional_allocation_analysis(merged_df):
    """
    Analyze detection rates using proportional allocation approach
    """
    
    print(f"\n=== PROPORTIONAL ALLOCATION ANALYSIS ===")
    
    detection_credits = []
    
    for acc in merged_df['ACCESSION_NUMBER'].unique():
        acc_data = merged_df[merged_df['ACCESSION_NUMBER'] == acc]
        
        # Count cancer type frequencies in this accession
        cancer_counts = acc_data['cancer_type'].value_counts()
        total_cancer_instances = len(acc_data)  # Total rows for this accession
        
        # Count total detections in this accession
        total_detections = sum(acc_data['caliper_boxes'].apply(is_non_empty_caliper_boxes))
        
        # Allocate detections proportionally to each cancer type
        for cancer_type, count in cancer_counts.items():
            proportion = count / total_cancer_instances
            allocated_detections = proportion * total_detections
            detection_rate_per_instance = allocated_detections / count  # Detection credits per instance
            
            detection_credits.append({
                'accession': acc,
                'cancer_type': cancer_type,
                'cancer_instances': count,
                'total_instances': total_cancer_instances,
                'total_detections': total_detections,
                'proportion': proportion,
                'allocated_detections': allocated_detections,
                'detection_rate_per_instance': detection_rate_per_instance
            })
    
    # Convert to DataFrame
    credits_df = pd.DataFrame(detection_credits)
    
    # Calculate average detection rate by cancer type
    proportional_summary = credits_df.groupby('cancer_type').agg({
        'detection_rate_per_instance': ['mean', 'std', 'count'],
        'allocated_detections': 'sum',
        'cancer_instances': 'sum'
    }).round(3)
    
    # Flatten column names
    proportional_summary.columns = ['mean_detection_rate', 'std_detection_rate', 'n_accessions', 
                                   'total_allocated_detections', 'total_instances']
    
    return proportional_summary, credits_df

def probabilistic_em_assignment(merged_df, max_iterations=100, tolerance=1e-6):
    """
    Use Expectation-Maximization to learn detection probabilities for each cancer type
    """
    
    print(f"\n=== PROBABILISTIC EM ASSIGNMENT ===")
    
    # Get unique cancer types
    cancer_types = merged_df['cancer_type'].unique()
    n_cancer_types = len(cancer_types)
    cancer_to_idx = {ct: i for i, ct in enumerate(cancer_types)}
    
    # Initialize detection probabilities uniformly
    detection_probs = np.ones(n_cancer_types) * 0.5  # Start with 50% detection rate for all
    
    print(f"Initializing EM with {n_cancer_types} cancer types")
    
    # Prepare accession data
    accession_data = []
    for acc in merged_df['ACCESSION_NUMBER'].unique():
        acc_df = merged_df[merged_df['ACCESSION_NUMBER'] == acc]
        cancer_counts = acc_df['cancer_type'].value_counts()
        total_detections = sum(acc_df['caliper_boxes'].apply(is_non_empty_caliper_boxes))
        
        # Convert to cancer type vector
        cancer_vector = np.zeros(n_cancer_types)
        for cancer_type, count in cancer_counts.items():
            cancer_vector[cancer_to_idx[cancer_type]] = count
            
        accession_data.append({
            'accession': acc,
            'cancer_vector': cancer_vector,
            'total_detections': total_detections,
            'cancer_counts': cancer_counts
        })
    
    # EM Algorithm
    log_likelihood_history = []
    
    for iteration in range(max_iterations):
        # E-step: Calculate expected assignments
        total_expected_detections = np.zeros(n_cancer_types)
        total_cancer_instances = np.zeros(n_cancer_types)
        log_likelihood = 0
        
        for acc_data in accession_data:
            cancer_vector = acc_data['cancer_vector']
            total_detections = acc_data['total_detections']
            
            # Expected detections for each cancer type in this accession
            expected_detections_per_instance = detection_probs
            expected_total_detections = np.sum(cancer_vector * expected_detections_per_instance)
            
            if expected_total_detections > 0:
                # Scale to match actual detections
                scaling_factor = total_detections / expected_total_detections
                expected_detections = cancer_vector * expected_detections_per_instance * scaling_factor
            else:
                expected_detections = np.zeros(n_cancer_types)
            
            # Accumulate for M-step
            total_expected_detections += expected_detections
            total_cancer_instances += cancer_vector
            
            # Calculate log-likelihood contribution (simplified)
            if total_detections > 0 and expected_total_detections > 0:
                log_likelihood += total_detections * np.log(expected_total_detections) - expected_total_detections
        
        log_likelihood_history.append(log_likelihood)
        
        # M-step: Update detection probabilities
        new_detection_probs = np.divide(total_expected_detections, total_cancer_instances, 
                                       out=np.zeros_like(total_expected_detections), 
                                       where=total_cancer_instances!=0)
        
        # Ensure probabilities are in valid range
        new_detection_probs = np.clip(new_detection_probs, 0.001, 0.999)
        
        # Check convergence
        change = np.max(np.abs(new_detection_probs - detection_probs))
        
        if change < tolerance:
            print(f"✓ EM converged after {iteration+1} iterations")
            break
            
        detection_probs = new_detection_probs
    
    # Final assignment using learned probabilities
    em_results = []
    for acc_data in accession_data:
        cancer_vector = acc_data['cancer_vector']
        total_detections = acc_data['total_detections']
        
        # Calculate expected detections for each cancer type
        expected_detections_per_instance = detection_probs
        expected_total = np.sum(cancer_vector * expected_detections_per_instance)
        
        if expected_total > 0:
            scaling_factor = total_detections / expected_total
            final_assignments = cancer_vector * expected_detections_per_instance * scaling_factor
        else:
            final_assignments = np.zeros(n_cancer_types)
        
        # Record results for each cancer type present
        for cancer_type, count in acc_data['cancer_counts'].items():
            if count > 0:
                idx = cancer_to_idx[cancer_type]
                detection_rate = final_assignments[idx] / count
                
                em_results.append({
                    'accession': acc_data['accession'],
                    'cancer_type': cancer_type,
                    'cancer_instances': count,
                    'expected_detections': final_assignments[idx],
                    'detection_rate': detection_rate,
                    'learned_prob': detection_probs[idx]
                })
    
    em_df = pd.DataFrame(em_results)
    
    # Create summary by cancer type
    em_summary = em_df.groupby('cancer_type').agg({
        'detection_rate': ['mean', 'std', 'count'],
        'expected_detections': 'sum',
        'cancer_instances': 'sum',
        'learned_prob': 'first'
    }).round(3)
    
    em_summary.columns = ['mean_detection_rate', 'std_detection_rate', 'n_accessions', 
                         'total_expected_detections', 'total_instances', 'learned_probability']
    
    return em_summary, em_df, detection_probs, cancer_types, log_likelihood_history

def optimal_assignment(merged_df, detection_probs, cancer_types):
    """
    Use optimal assignment (Hungarian algorithm) to assign detections to cancer types
    """
    
    print(f"\n=== OPTIMAL ASSIGNMENT ===")
    
    cancer_to_idx = {ct: i for i, ct in enumerate(cancer_types)}
    
    optimal_results = []
    
    for acc in merged_df['ACCESSION_NUMBER'].unique():
        acc_df = merged_df[merged_df['ACCESSION_NUMBER'] == acc]
        cancer_counts = acc_df['cancer_type'].value_counts()
        total_detections = sum(acc_df['caliper_boxes'].apply(is_non_empty_caliper_boxes))
        
        # Create list of cancer type instances for this accession
        cancer_instances = []
        for cancer_type, count in cancer_counts.items():
            cancer_instances.extend([cancer_type] * count)
        
        if len(cancer_instances) == 0 or total_detections == 0:
            continue
            
        # Create cost matrix for assignment
        n_instances = len(cancer_instances)
        n_assignments = max(n_instances, total_detections)
        
        # Cost matrix: lower cost = higher probability of detection
        cost_matrix = np.zeros((n_assignments, n_assignments))
        
        for i in range(n_instances):
            cancer_type = cancer_instances[i]
            cancer_idx = cancer_to_idx[cancer_type]
            detection_prob = detection_probs[cancer_idx]
            
            # Lower cost for higher detection probability
            cost_matrix[i, :total_detections] = 1 - detection_prob
            
            # High cost for not assigning detection
            if total_detections < n_assignments:
                cost_matrix[i, total_detections:] = 1.0
        
        # Fill remaining rows (if more detections than instances)
        if total_detections > n_instances:
            cost_matrix[n_instances:, :] = 0.5
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Extract assignments
        assignments = {}
        for cancer_type in cancer_counts.keys():
            assignments[cancer_type] = 0
            
        for row, col in zip(row_indices, col_indices):
            if row < len(cancer_instances) and col < total_detections:
                cancer_type = cancer_instances[row]
                assignments[cancer_type] += 1
        
        # Record results
        for cancer_type, count in cancer_counts.items():
            assigned_detections = assignments.get(cancer_type, 0)
            detection_rate = assigned_detections / count if count > 0 else 0
            
            optimal_results.append({
                'accession': acc,
                'cancer_type': cancer_type,
                'cancer_instances': count,
                'assigned_detections': assigned_detections,
                'detection_rate': detection_rate,
                'total_detections': total_detections
            })
    
    optimal_df = pd.DataFrame(optimal_results)
    
    # Create summary by cancer type
    optimal_summary = optimal_df.groupby('cancer_type').agg({
        'detection_rate': ['mean', 'std', 'count'],
        'assigned_detections': 'sum',
        'cancer_instances': 'sum'
    }).round(3)
    
    optimal_summary.columns = ['mean_detection_rate', 'std_detection_rate', 'n_accessions', 
                              'total_assigned_detections', 'total_instances']
    
    return optimal_summary, optimal_df

def apply_sample_size_filter(summary_df, min_accessions, min_instances):
    """Apply sample size filtering to summary dataframe"""
    
    meets_accession_threshold = summary_df['n_accessions'] >= min_accessions
    meets_instance_threshold = summary_df['total_instances'] >= min_instances
    meets_both_thresholds = meets_accession_threshold & meets_instance_threshold
    
    filtered_df = summary_df[meets_both_thresholds].copy()
    filtered_df = filtered_df.sort_values('mean_detection_rate', ascending=True)
    
    return filtered_df

def create_comprehensive_visualization(proportional_summary, em_summary, optimal_summary, 
                                     log_likelihood_history, min_accessions, min_instances):
    """Create comprehensive 4-panel visualization comparing all methods"""
    
    # Apply filtering to all methods
    filtered_proportional = apply_sample_size_filter(proportional_summary, min_accessions, min_instances)
    filtered_em = apply_sample_size_filter(em_summary, min_accessions, min_instances)
    filtered_optimal = apply_sample_size_filter(optimal_summary, min_accessions, min_instances)
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 18))
    
    # 1. Proportional Allocation Results
    if len(filtered_proportional) > 0:
        prop_data = filtered_proportional.reset_index()
        bars1 = ax1.bar(range(len(prop_data)), prop_data['mean_detection_rate'], 
                       color='lightgreen', alpha=0.7, edgecolor='darkgreen')
        ax1.errorbar(range(len(prop_data)), prop_data['mean_detection_rate'], 
                    yerr=prop_data['std_detection_rate'], fmt='none', color='black', alpha=0.5)
        
        ax1.set_xlabel('Cancer Type', fontsize=12)
        ax1.set_ylabel('Mean Detection Rate', fontsize=12)
        ax1.set_title(f'Proportional Allocation Method\n(≥{min_accessions} accessions, ≥{min_instances} instances)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(prop_data)))
        ax1.set_xticklabels(prop_data['cancer_type'], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, rate) in enumerate(zip(bars1, prop_data['mean_detection_rate'])):
            ax1.annotate(f'{rate:.3f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    else:
        ax1.text(0.5, 0.5, 'Insufficient data\nfor Proportional method', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=14)
        ax1.set_title('Proportional Allocation Method', fontsize=14, fontweight='bold')
    
    # 2. EM Algorithm Results
    if len(filtered_em) > 0:
        em_data = filtered_em.reset_index()
        bars2 = ax2.bar(range(len(em_data)), em_data['mean_detection_rate'], 
                       color='lightblue', alpha=0.7, edgecolor='navy')
        ax2.errorbar(range(len(em_data)), em_data['mean_detection_rate'], 
                    yerr=em_data['std_detection_rate'], fmt='none', color='black', alpha=0.5)
        
        ax2.set_xlabel('Cancer Type', fontsize=12)
        ax2.set_ylabel('Mean Detection Rate', fontsize=12)
        ax2.set_title(f'EM Algorithm Method\n(≥{min_accessions} accessions, ≥{min_instances} instances)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(em_data)))
        ax2.set_xticklabels(em_data['cancer_type'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, rate) in enumerate(zip(bars2, em_data['mean_detection_rate'])):
            ax2.annotate(f'{rate:.3f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'Insufficient data\nfor EM method', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('EM Algorithm Method', fontsize=14, fontweight='bold')
    
    # 3. Optimal Assignment Results
    if len(filtered_optimal) > 0:
        optimal_data = filtered_optimal.reset_index()
        bars3 = ax3.bar(range(len(optimal_data)), optimal_data['mean_detection_rate'], 
                       color='lightcoral', alpha=0.7, edgecolor='darkred')
        ax3.errorbar(range(len(optimal_data)), optimal_data['mean_detection_rate'], 
                    yerr=optimal_data['std_detection_rate'], fmt='none', color='black', alpha=0.5)
        
        ax3.set_xlabel('Cancer Type', fontsize=12)
        ax3.set_ylabel('Mean Detection Rate', fontsize=12)
        ax3.set_title(f'Optimal Assignment Method\n(≥{min_accessions} accessions, ≥{min_instances} instances)', 
                     fontsize=14, fontweight='bold')
        ax3.set_xticks(range(len(optimal_data)))
        ax3.set_xticklabels(optimal_data['cancer_type'], rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, rate) in enumerate(zip(bars3, optimal_data['mean_detection_rate'])):
            ax3.annotate(f'{rate:.3f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    else:
        ax3.text(0.5, 0.5, 'Insufficient data\nfor Optimal method', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        ax3.set_title('Optimal Assignment Method', fontsize=14, fontweight='bold')
    
    # 4. Method Comparison with Average - Enhanced version
    all_methods_data = []
    
    if len(filtered_proportional) > 0 and len(filtered_em) > 0 and len(filtered_optimal) > 0:
        # Find cancer types present in all three methods - convert set to list
        overlap = (set(filtered_proportional.index) & 
                  set(filtered_em.index) & 
                  set(filtered_optimal.index))
        overlap_list = list(overlap)  # Convert set to list for pandas indexing
        
        if len(overlap_list) > 0:
            # Create comparison data with average calculation
            for cancer_type in overlap_list:
                prop_rate = filtered_proportional.loc[cancer_type, 'mean_detection_rate']
                em_rate = filtered_em.loc[cancer_type, 'mean_detection_rate']
                opt_rate = filtered_optimal.loc[cancer_type, 'mean_detection_rate']
                avg_rate = (prop_rate + em_rate + opt_rate) / 3
                
                all_methods_data.append({
                    'cancer_type': cancer_type,
                    'proportional': prop_rate,
                    'em': em_rate,
                    'optimal': opt_rate,
                    'average': avg_rate
                })
            
            comparison_df = pd.DataFrame(all_methods_data)
            # Sort by average detection rate (lowest first - most problematic)
            comparison_df = comparison_df.sort_values('average').reset_index(drop=True)
            
            # Create grouped bar chart with 4 bars
            x_pos = np.arange(len(comparison_df))
            width = 0.2  # Narrower bars to fit 4 groups
            
            bars4a = ax4.bar(x_pos - 1.5*width, comparison_df['proportional'], width, 
                           label='Proportional', color='lightgreen', alpha=0.7)
            bars4b = ax4.bar(x_pos - 0.5*width, comparison_df['em'], width, 
                           label='EM Algorithm', color='lightblue', alpha=0.7)
            bars4c = ax4.bar(x_pos + 0.5*width, comparison_df['optimal'], width,
                           label='Optimal Assignment', color='lightcoral', alpha=0.7)
            bars4d = ax4.bar(x_pos + 1.5*width, comparison_df['average'], width,
                           label='Average', color='gold', alpha=0.8, edgecolor='orange', linewidth=2)
            
            ax4.set_xlabel('Cancer Type', fontsize=12)
            ax4.set_ylabel('Mean Detection Rate', fontsize=12)
            ax4.set_title('Method Comparison\n(Sorted by Average Detection Rate)', 
                         fontsize=14, fontweight='bold')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(comparison_df['cancer_type'], rotation=45, ha='right')
            ax4.legend(loc='upper left')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels for average bars only (to avoid clutter)
            for i, (bar, avg_rate) in enumerate(zip(bars4d, comparison_df['average'])):
                ax4.annotate(f'{avg_rate:.3f}', 
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 5), textcoords="offset points", 
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            print(f"Comparison includes {len(overlap_list)} cancer types present in all methods")
            print("Cancer types sorted by average detection rate (lowest first):")
            for _, row in comparison_df.iterrows():
                print(f"  {row['cancer_type']}: Avg={row['average']:.3f} "
                      f"(Prop={row['proportional']:.3f}, EM={row['em']:.3f}, Opt={row['optimal']:.3f})")
        else:
            ax4.text(0.5, 0.5, 'No cancer types\nmeet criteria in all methods', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=14)
            ax4.set_title('Method Comparison', fontsize=14, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Insufficient data\nfor method comparison', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Method Comparison', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('comprehensive_method_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Also create EM convergence plot separately
    if len(log_likelihood_history) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(log_likelihood_history, 'b-', linewidth=2)
        plt.title('EM Algorithm Convergence', fontsize=14, fontweight='bold')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Log-Likelihood', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig('em_convergence.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print("✓ Comprehensive visualization saved as 'comprehensive_method_comparison.png'")
    print("✓ EM convergence plot saved as 'em_convergence.png'")
    
    return all_methods_data

# Main execution
if __name__ == "__main__":
    # File paths - UPDATE THESE WITH YOUR ACTUAL PATHS
    lesion_csv_path = r"C:\Users\Tristan\Desktop\Databases_database_2025_9_10_main_LesionData.csv"
    image_csv_path = r"C:\Users\Tristan\Desktop\Databases_database_2025_8_11_main_ImageData.csv"
    
    # SAMPLE SIZE THRESHOLDS - ADJUST THESE AS NEEDED
    MIN_ACCESSIONS = 15    # Minimum accessions containing this cancer type
    MIN_INSTANCES = 50     # Minimum total instances across all accessions
    
    # Run the analysis
    try:
        # Load and merge data
        merged_df = analyze_cancer_distributions(lesion_csv_path, image_csv_path)
        
        # 1. Proportional Allocation Analysis
        proportional_summary, proportional_df = proportional_allocation_analysis(merged_df)
        
        # 2. Probabilistic EM Assignment
        em_summary, em_df, detection_probs, cancer_types, log_likelihood_history = probabilistic_em_assignment(merged_df)
        
        # 3. Optimal Assignment using EM probabilities
        optimal_summary, optimal_df = optimal_assignment(merged_df, detection_probs, cancer_types)
        
        # Create comprehensive visualization
        comparison_data = create_comprehensive_visualization(
            proportional_summary, em_summary, optimal_summary, 
            log_likelihood_history, MIN_ACCESSIONS, MIN_INSTANCES
        )
        
        # Apply sample size filtering for detailed analysis
        filtered_proportional = apply_sample_size_filter(proportional_summary, MIN_ACCESSIONS, MIN_INSTANCES)
        filtered_em = apply_sample_size_filter(em_summary, MIN_ACCESSIONS, MIN_INSTANCES)
        filtered_optimal = apply_sample_size_filter(optimal_summary, MIN_ACCESSIONS, MIN_INSTANCES)
        
        # Save all results
        proportional_summary.to_csv('proportional_allocation_results.csv', index=True)
        em_summary.to_csv('em_assignment_results.csv', index=True)
        optimal_summary.to_csv('optimal_assignment_results.csv', index=True)
        
        filtered_proportional.to_csv('proportional_allocation_filtered.csv', index=True)
        filtered_em.to_csv('em_assignment_filtered.csv', index=True)
        filtered_optimal.to_csv('optimal_assignment_filtered.csv', index=True)
        
        # Save detailed results
        proportional_df.to_csv('proportional_detailed_results.csv', index=False)
        em_df.to_csv('em_detailed_results.csv', index=False)
        optimal_df.to_csv('optimal_detailed_results.csv', index=False)
        
        # Save learned detection probabilities
        prob_df = pd.DataFrame({
            'cancer_type': cancer_types,
            'learned_detection_probability': detection_probs
        }).sort_values('learned_detection_probability')
        prob_df.to_csv('learned_detection_probabilities.csv', index=False)
        
        print("✓ All results saved with appropriate suffixes")
        print("✓ Learned probabilities saved as 'learned_detection_probabilities.csv'")
        
        # Print comprehensive insights - Fixed set indexing
        print("\n" + "="*60)
        print("=== COMPREHENSIVE METHOD COMPARISON ===")
        print("="*60)
        
        print(f"\n--- Sample Size Summary ---")
        print(f"Proportional method: {len(filtered_proportional)}/{len(proportional_summary)} cancer types meet thresholds")
        print(f"EM method: {len(filtered_em)}/{len(em_summary)} cancer types meet thresholds")
        print(f"Optimal method: {len(filtered_optimal)}/{len(optimal_summary)} cancer types meet thresholds")
        
        # Show top/bottom performers for each method
        for method_name, filtered_data in [("Proportional", filtered_proportional), 
                                          ("EM", filtered_em), 
                                          ("Optimal", filtered_optimal)]:
            print(f"\n--- {method_name} Method Results (Filtered) ---")
            if len(filtered_data) > 0:
                print("Lowest detection rates:")
                worst = filtered_data.head(3)
                for cancer_type, row in worst.iterrows():
                    print(f"  {cancer_type}: {row['mean_detection_rate']:.3f}")
                
                print("Highest detection rates:")
                best = filtered_data.tail(3)
                for cancer_type, row in best.iterrows():
                    print(f"  {cancer_type}: {row['mean_detection_rate']:.3f}")
            else:
                print("  No cancer types meet sample size requirements")
        
        # Method correlation analysis - Fixed set to list conversion
        if len(comparison_data) > 0:
            print(f"\n--- Cross-Method Analysis ---")
            comp_df = pd.DataFrame(comparison_data)
            
            # Calculate correlations
            corr_prop_em = np.corrcoef(comp_df['proportional'], comp_df['em'])[0, 1]
            corr_prop_opt = np.corrcoef(comp_df['proportional'], comp_df['optimal'])[0, 1]
            corr_em_opt = np.corrcoef(comp_df['em'], comp_df['optimal'])[0, 1]
            
            print(f"Correlation - Proportional vs EM: {corr_prop_em:.3f}")
            print(f"Correlation - Proportional vs Optimal: {corr_prop_opt:.3f}")
            print(f"Correlation - EM vs Optimal: {corr_em_opt:.3f}")
            
            print(f"\nCancer types analyzed by all methods: {len(comp_df)}")
            for _, row in comp_df.iterrows():
                print(f"  {row['cancer_type']}: Prop={row['proportional']:.3f}, "
                      f"EM={row['em']:.3f}, Opt={row['optimal']:.3f}")
        else:
            print(f"\n--- Cross-Method Analysis ---")
            # Fix the set indexing error by converting to list
            if len(filtered_em) > 0 and len(filtered_optimal) > 0:
                overlap = set(filtered_em.index) & set(filtered_optimal.index)
                overlap_list = list(overlap)  # Convert set to list
                if overlap_list:
                    print(f"Cancer types analyzed by both methods: {len(overlap_list)}")
                    
                    # Calculate correlation
                    em_rates = filtered_em.loc[overlap_list, 'mean_detection_rate']
                    optimal_rates = filtered_optimal.loc[overlap_list, 'mean_detection_rate']
                    correlation = np.corrcoef(em_rates, optimal_rates)[0, 1]
                    print(f"Correlation between methods: {correlation:.3f}")
                else:
                    print("No overlapping cancer types between methods (sample size limitations)")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()