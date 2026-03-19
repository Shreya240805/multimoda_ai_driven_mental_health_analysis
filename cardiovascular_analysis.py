"""
Cardiovascular Analysis Module
"""
import pandas as pd
import numpy as np

def load_heart_rate_data(csv_path):
    """Load Samsung Health heart rate CSV"""
    try:
        df = pd.read_csv(csv_path)
        
        if 'heart_rate' not in df.columns:
            for col in df.columns:
                if 'heart' in col.lower() or 'bpm' in col.lower():
                    df['heart_rate'] = df[col]
                    break
        
        df = df.dropna(subset=['heart_rate'])
        df['heart_rate'] = pd.to_numeric(df['heart_rate'], errors='coerce')
        df = df.dropna(subset=['heart_rate'])
        
        return df
    except Exception as e:
        print(f"Error loading heart rate data: {e}")
        return None


def calculate_hrv_metrics(heart_rates):
    """Calculate HRV metrics"""
    if len(heart_rates) < 2:
        return None
    
    rr_intervals = 60000 / heart_rates
    successive_diffs = np.diff(rr_intervals)
    
    sdnn = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(successive_diffs ** 2))
    nn50 = np.sum(np.abs(successive_diffs) > 50)
    pnn50 = (nn50 / len(successive_diffs)) * 100 if len(successive_diffs) > 0 else 0
    
    return {
        'sdnn': float(sdnn),
        'rmssd': float(rmssd),
        'pnn50': float(pnn50),
        'mean_hr': float(np.mean(heart_rates)),
        'max_hr': float(np.max(heart_rates)),
        'min_hr': float(np.min(heart_rates)),
        'hr_range': float(np.max(heart_rates) - np.min(heart_rates))
    }


def analyze_cardiovascular_stress(csv_path):
    """Analyze cardiovascular stress from CSV"""
    df = load_heart_rate_data(csv_path)
    
    if df is None or len(df) == 0:
        return None
    
    heart_rates = df['heart_rate'].values
    metrics = calculate_hrv_metrics(heart_rates)
    
    if metrics is None:
        return None
    
    stress_score = 0
    stress_indicators = []
    
    if metrics['mean_hr'] > 90:
        stress_score += 3
        stress_indicators.append(f"Elevated mean HR ({metrics['mean_hr']:.1f} BPM)")
    elif metrics['mean_hr'] > 80:
        stress_score += 2
    
    if metrics['rmssd'] < 15:
        stress_score += 3
        stress_indicators.append(f"Very low HRV (RMSSD: {metrics['rmssd']:.1f}ms)")
    elif metrics['rmssd'] < 25:
        stress_score += 2
    
    if metrics['sdnn'] < 40:
        stress_score += 2
        stress_indicators.append(f"Low SDNN ({metrics['sdnn']:.1f}ms)")
    
    if stress_score >= 7:
        stress_level = 'High'
    elif stress_score >= 4:
        stress_level = 'Moderate'
    else:
        stress_level = 'Low'
    
    return {
        'metrics': metrics,
        'stress_score': stress_score,
        'stress_level': stress_level,
        'stress_indicators': stress_indicators,
        'data_points': len(heart_rates)
    }
