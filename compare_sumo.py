#!/usr/bin/env python3
"""Compare FLAMEGPU simulation output with SUMO edgeData output."""

import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

def parse_sumo_edgedata(xml_path: str, max_time: float = None) -> pd.DataFrame:
    """Parse SUMO edgeData XML file into a DataFrame."""
    print(f"Parsing SUMO edgeData from {xml_path}...")
    
    records = []
    context = ET.iterparse(xml_path, events=('end',))
    
    for event, elem in context:
        if elem.tag == 'interval':
            begin = float(elem.get('begin', 0))
            end = float(elem.get('end', 0))
            
            if max_time and begin >= max_time:
                elem.clear()
                continue
            
            for edge in elem.findall('edge'):
                records.append({
                    'interval_begin': begin,
                    'interval_end': end,
                    'id': edge.get('id'),
                    'sampledSeconds': float(edge.get('sampledSeconds', 0)),
                    'traveltime': float(edge.get('traveltime', 0)),
                    'speed': float(edge.get('speed', 0)),
                    'speedRelative': float(edge.get('speedRelative', 1)),
                    'entered': int(edge.get('entered', 0)),
                    'left': int(edge.get('left', 0)),
                    'departed': int(edge.get('departed', 0)),
                    'arrived': int(edge.get('arrived', 0)),
                })
            elem.clear()
    
    df = pd.DataFrame(records)
    print(f"  Parsed {len(df)} records")
    return df


def compare_metrics(sumo_df: pd.DataFrame, flamegpu_df: pd.DataFrame) -> dict:
    """Compare metrics between SUMO and FLAMEGPU outputs."""
    
    # Get common intervals
    sumo_intervals = set(sumo_df['interval_begin'].unique())
    fg_intervals = set(flamegpu_df['interval_begin'].unique())
    common_intervals = sorted(sumo_intervals & fg_intervals)
    
    print(f"\nSUMO intervals: {sorted(sumo_intervals)}")
    print(f"FLAMEGPU intervals: {sorted(fg_intervals)}")
    print(f"Common intervals: {common_intervals}")
    
    results = {
        'per_interval': [],
        'per_edge': [],
    }
    
    # Per-interval comparison
    print("\n" + "="*80)
    print("PER-INTERVAL COMPARISON (Network Totals)")
    print("="*80)
    print(f"{'Interval':<15} {'Metric':<15} {'SUMO':>12} {'FLAMEGPU':>12} {'Diff':>10} {'Diff%':>10}")
    print("-"*80)
    
    for interval in common_intervals:
        sumo_int = sumo_df[sumo_df['interval_begin'] == interval]
        fg_int = flamegpu_df[flamegpu_df['interval_begin'] == interval]
        
        metrics = {
            'entered': (sumo_int['entered'].sum(), fg_int['entered'].sum()),
            'left': (sumo_int['left'].sum(), fg_int['left'].sum()),
            'sampledSec': (sumo_int['sampledSeconds'].sum(), fg_int['sampledSeconds'].sum()),
        }
        
        interval_str = f"{interval:.0f}-{interval+900:.0f}"
        
        for metric_name, (sumo_val, fg_val) in metrics.items():
            diff = fg_val - sumo_val
            diff_pct = (diff / sumo_val * 100) if sumo_val != 0 else 0
            print(f"{interval_str:<15} {metric_name:<15} {sumo_val:>12.0f} {fg_val:>12.0f} {diff:>+10.0f} {diff_pct:>+9.1f}%")
            
            results['per_interval'].append({
                'interval': interval,
                'metric': metric_name,
                'sumo': sumo_val,
                'flamegpu': fg_val,
                'diff': diff,
                'diff_pct': diff_pct,
            })
    
    # Per-edge comparison for first interval (sampled)
    print("\n" + "="*80)
    print("PER-EDGE COMPARISON (First interval, edges with activity)")
    print("="*80)
    
    first_interval = common_intervals[0]
    sumo_first = sumo_df[sumo_df['interval_begin'] == first_interval].copy()
    fg_first = flamegpu_df[flamegpu_df['interval_begin'] == first_interval].copy()
    
    # Get common edges with activity
    sumo_active = sumo_first[sumo_first['sampledSeconds'] > 0]['id'].unique()
    fg_active = fg_first[fg_first['sampledSeconds'] > 0]['id'].unique()
    common_edges = set(sumo_active) & set(fg_active)
    
    print(f"SUMO active edges: {len(sumo_active)}")
    print(f"FLAMEGPU active edges: {len(fg_active)}")
    print(f"Common active edges: {len(common_edges)}")
    
    if common_edges:
        # Merge on edge id
        merged = sumo_first[sumo_first['id'].isin(common_edges)].merge(
            fg_first[fg_first['id'].isin(common_edges)],
            on='id',
            suffixes=('_sumo', '_fg')
        )
        
        print(f"\nSample of edge-level comparison (first 10 edges with biggest differences):")
        print(f"{'Edge ID':<25} {'sampledSec_S':>12} {'sampledSec_F':>12} {'Diff%':>10} {'entered_S':>10} {'entered_F':>10}")
        print("-"*90)
        
        merged['sampledSec_diff_pct'] = np.abs(
            (merged['sampledSeconds_fg'] - merged['sampledSeconds_sumo']) / 
            merged['sampledSeconds_sumo'].replace(0, 1) * 100
        )
        
        top_diff = merged.nlargest(10, 'sampledSec_diff_pct')
        for _, row in top_diff.iterrows():
            diff_pct = ((row['sampledSeconds_fg'] - row['sampledSeconds_sumo']) / 
                       row['sampledSeconds_sumo'] * 100) if row['sampledSeconds_sumo'] > 0 else 0
            print(f"{row['id']:<25} {row['sampledSeconds_sumo']:>12.1f} {row['sampledSeconds_fg']:>12.1f} "
                  f"{diff_pct:>+9.1f}% {row['entered_sumo']:>10} {row['entered_fg']:>10}")
        
        # Calculate overall correlation
        for metric in ['sampledSeconds', 'entered', 'left']:
            sumo_col = f'{metric}_sumo'
            fg_col = f'{metric}_fg'
            if sumo_col in merged.columns and fg_col in merged.columns:
                correlation = merged[sumo_col].corr(merged[fg_col])
                mae = np.abs(merged[sumo_col] - merged[fg_col]).mean()
                print(f"\n{metric}: Correlation = {correlation:.4f}, MAE = {mae:.2f}")
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    total_sumo_entered = sumo_df[sumo_df['interval_begin'].isin(common_intervals)]['entered'].sum()
    total_fg_entered = flamegpu_df[flamegpu_df['interval_begin'].isin(common_intervals)]['entered'].sum()
    total_sumo_left = sumo_df[sumo_df['interval_begin'].isin(common_intervals)]['left'].sum()
    total_fg_left = flamegpu_df[flamegpu_df['interval_begin'].isin(common_intervals)]['left'].sum()
    total_sumo_sampled = sumo_df[sumo_df['interval_begin'].isin(common_intervals)]['sampledSeconds'].sum()
    total_fg_sampled = flamegpu_df[flamegpu_df['interval_begin'].isin(common_intervals)]['sampledSeconds'].sum()
    
    print(f"{'Metric':<20} {'SUMO':>15} {'FLAMEGPU':>15} {'Diff':>12} {'Diff%':>10}")
    print("-"*72)
    
    for name, sumo_val, fg_val in [
        ('Total entered', total_sumo_entered, total_fg_entered),
        ('Total left', total_sumo_left, total_fg_left),
        ('Total sampledSec', total_sumo_sampled, total_fg_sampled),
    ]:
        diff = fg_val - sumo_val
        diff_pct = (diff / sumo_val * 100) if sumo_val != 0 else 0
        print(f"{name:<20} {sumo_val:>15.0f} {fg_val:>15.0f} {diff:>+12.0f} {diff_pct:>+9.1f}%")
    
    return results


def main():
    # Paths
    sumo_xml = Path("/home/prnvstsh/flamegpu_traffic_mesoscopic/arbon/edge_data_arbon.out.xml")
    flamegpu_parquet = Path("/home/prnvstsh/flamegpu_traffic_mesoscopic/arbon_metrics_edges.parquet")
    
    # Parse data
    flamegpu_df = pd.read_parquet(flamegpu_parquet)
    print(f"Loaded FLAMEGPU data: {len(flamegpu_df)} records")
    print(f"  Intervals: {sorted(flamegpu_df['interval_begin'].unique())}")
    
    max_time = flamegpu_df['interval_end'].max()
    print(f"  Max time: {max_time}s")
    
    sumo_df = parse_sumo_edgedata(sumo_xml, max_time=max_time)
    
    # Compare
    results = compare_metrics(sumo_df, flamegpu_df)
    
    # Save comparison results
    interval_df = pd.DataFrame(results['per_interval'])
    interval_df.to_csv('/home/prnvstsh/flamegpu_traffic_mesoscopic/comparison_results.csv', index=False)
    print(f"\nComparison saved to comparison_results.csv")


if __name__ == "__main__":
    main()

