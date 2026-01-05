"""
preprocessing.py
Column mapping, standardization, and validation logic
"""

import pandas as pd
import numpy as np


# Mapping keywords for automatic column detection
COLUMN_MAPPING_KEYWORDS = {
    'absences': ['absences', 'absence', 'absent', 'attendance'],
    'studytime': ['studytime', 'study_time', 'study_hours', 'study', 'hours'],
    'failures': ['failures', 'failure', 'past_failures', 'failed', 'fails']
}


def auto_map_columns(df):
    """
    Attempts to automatically map uploaded CSV columns to required features
    
    Args:
        df: pandas DataFrame with user's column names
    
    Returns:
        dict: Mapping from required columns to uploaded columns
              Example: {'absences': 'Attendance', 'studytime': 'study_hours'}
    """
    mapping = {}
    df_columns_lower = {col: col for col in df.columns}
    
    for required_col, keywords in COLUMN_MAPPING_KEYWORDS.items():
        for user_col in df.columns:
            user_col_lower = user_col.lower()
            # Check if any keyword matches
            if any(keyword in user_col_lower for keyword in keywords):
                mapping[required_col] = user_col
                break
    
    return mapping


def standardize_dataframe(df, mapping):
    """
    Converts uploaded DataFrame into standardized internal schema
    
    Args:
        df: Original uploaded DataFrame
        mapping: Column mapping dict from auto_map_columns
    
    Returns:
        pandas DataFrame with standardized columns
    """
    standardized = pd.DataFrame()
    
    # Map the columns
    for required_col, user_col in mapping.items():
        if user_col in df.columns:
            standardized[required_col] = df[user_col]
    
    # Add missing columns with default values if needed
    required_cols = ['absences', 'studytime', 'failures']
    for col in required_cols:
        if col not in standardized.columns:
            standardized[col] = 0
    
    # Convert to numeric, coerce errors to NaN
    for col in required_cols:
        standardized[col] = pd.to_numeric(standardized[col], errors='coerce')
    
    # Create simulated target variable for demo
    # In production, this would come from actual pass/fail data
    standardized['pass_fail'] = (standardized['failures'] == 0).astype(int)
    
    # Add student ID if not present
    if 'student_id' not in standardized.columns:
        standardized.insert(0, 'student_id', range(1, len(standardized) + 1))
    
    return standardized


def validate_dataframe(df):
    """
    Validates that DataFrame is ready for model training
    
    Args:
        df: Standardized DataFrame
    
    Returns:
        tuple: (is_valid: bool, error_messages: list)
    """
    errors = []
    required_cols = ['absences', 'studytime', 'failures']
    
    # Check for required columns
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    
    # Check for missing values
    if df[required_cols].isnull().any().any():
        null_cols = df[required_cols].columns[df[required_cols].isnull().any()].tolist()
        errors.append(f"Missing values found in columns: {', '.join(null_cols)}")
    
    # Check for negative values
    for col in required_cols:
        if col in df.columns and (df[col] < 0).any():
            errors.append(f"Negative values found in column: {col}")
    
    # Check minimum number of rows
    if len(df) < 5:
        errors.append("Dataset must contain at least 5 students")
    
    is_valid = len(errors) == 0
    return is_valid, errors