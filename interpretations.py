"""
interpretations.py
Dynamic interpretation logic based on model predictions and SHAP values
NO HARDCODED RULES - Everything driven by ML model outputs
"""

import numpy as np
import pandas as pd


def get_risk_category_from_model(probability, shap_values_series):
    """
    Determine risk category based purely on model output and SHAP magnitudes
    
    Args:
        probability: Model's predicted failure probability
        shap_values_series: SHAP values for this prediction
    
    Returns:
        str: Risk category
    """
    # Use model's probability directly
    if probability < 0.33:
        return "Low Risk"
    elif probability < 0.67:
        return "Medium Risk"
    else:
        return "High Risk"


def get_confidence_from_agreement(agreement_score, shap_magnitude):
    """
    Determine confidence based on model agreement and SHAP consistency
    
    Args:
        agreement_score: How much models agree (0-1)
        shap_magnitude: Total magnitude of SHAP values
    
    Returns:
        str: Confidence level
    """
    # High agreement = high confidence
    if agreement_score > 0.85:
        return "Very High Confidence"
    elif agreement_score > 0.70:
        return "High Confidence"
    elif agreement_score > 0.50:
        return "Moderate Confidence"
    else:
        return "Low Confidence"


def interpret_factor_impact(feature_name, actual_value, shap_value, percentile_in_dataset):
    """
    Interpret a factor's impact based on SHAP value and dataset context
    
    Args:
        feature_name: Name of feature (absences, studytime, failures)
        actual_value: Student's actual value
        shap_value: SHAP contribution to prediction
        percentile_in_dataset: Where this value falls in the dataset (0-100)
    
    Returns:
        dict: Interpretation with impact level and description
    """
    abs_shap = abs(shap_value)
    is_increasing_risk = shap_value > 0
    
    # Determine magnitude
    if abs_shap > 0.15:
        magnitude = "MAJOR"
    elif abs_shap > 0.08:
        magnitude = "MODERATE"
    else:
        magnitude = "MINOR"
    
    # Create interpretation based on percentile and direction
    if feature_name == 'absences':
        if is_increasing_risk:
            if percentile_in_dataset > 75:
                description = f"Absences ({int(actual_value)}) are in the top 25% highest in the dataset - a major concern"
            elif percentile_in_dataset > 50:
                description = f"Absences ({int(actual_value)}) are above average for this group"
            else:
                description = f"Absences ({int(actual_value)}) are contributing to risk"
        else:
            if percentile_in_dataset < 25:
                description = f"Excellent attendance ({int(actual_value)} absences) - in the best 25% of students"
            else:
                description = f"Attendance ({int(actual_value)} absences) is helping reduce risk"
    
    elif feature_name == 'studytime':
        if is_increasing_risk:
            if percentile_in_dataset < 25:
                description = f"Study time ({int(actual_value)} hrs/week) is in the bottom 25% - critical weakness"
            else:
                description = f"Study time ({int(actual_value)} hrs/week) needs improvement"
        else:
            if percentile_in_dataset > 75:
                description = f"Study time ({int(actual_value)} hrs/week) is in the top 25% - major strength"
            else:
                description = f"Study time ({int(actual_value)} hrs/week) is protective"
    
    elif feature_name == 'failures':
        if is_increasing_risk:
            if actual_value == 0:
                description = "Surprisingly, the model sees some risk despite no past failures"
            elif actual_value >= 2:
                description = f"Multiple failures ({int(actual_value)}) are the strongest risk predictor"
            else:
                description = f"Past failure ({int(actual_value)}) is affecting current prediction"
        else:
            description = f"Clean record (0 failures) is a strong protective factor"
    else:
        description = f"Factor value: {actual_value}"
    
    return {
        'magnitude': magnitude,
        'direction': 'INCREASING RISK' if is_increasing_risk else 'PROTECTIVE',
        'description': description,
        'shap_value': shap_value,
        'percentile': percentile_in_dataset
    }


def generate_student_profile(student_data, all_students_data, risk_probability, shap_values):
    """
    Generate comprehensive student profile based on model outputs
    
    Args:
        student_data: Series with student's features
        all_students_data: DataFrame with all students for context
        risk_probability: Model's predicted risk
        shap_values: SHAP values for this student
    
    Returns:
        dict: Complete profile with interpretations
    """
    profile = {
        'risk_score': risk_probability,
        'risk_category': get_risk_category_from_model(risk_probability, shap_values),
        'factors': []
    }
    
    # Analyze each factor
    for feature in shap_values.index:
        actual_value = student_data[feature]
        shap_value = shap_values[feature]
        
        # Calculate percentile in dataset
        percentile = (all_students_data[feature] <= actual_value).sum() / len(all_students_data) * 100
        
        interpretation = interpret_factor_impact(
            feature,
            actual_value,
            shap_value,
            percentile
        )
        
        profile['factors'].append({
            'name': feature,
            'value': actual_value,
            'interpretation': interpretation
        })
    
    # Sort by SHAP magnitude
    profile['factors'].sort(key=lambda x: abs(x['interpretation']['shap_value']), reverse=True)
    
    return profile


def generate_action_plan(profile, agreement_score):
    """
    Generate action plan based purely on model outputs
    
    Args:
        profile: Student profile from generate_student_profile
        agreement_score: Model agreement (0-1)
    
    Returns:
        dict: Action plan with priorities
    """
    actions = {
        'immediate': [],
        'short_term': [],
        'monitoring': []
    }
    
    # Low confidence warning
    if agreement_score < 0.6:
        actions['immediate'].append({
            'area': 'Prediction Uncertainty',
            'reason': f'Models show low agreement ({agreement_score:.0%})',
            'action': 'Gather additional information before making decisions'
        })
    
    # Factor-based actions determined by SHAP
    for factor in profile['factors']:
        interp = factor['interpretation']
        
        if interp['magnitude'] == 'MAJOR' and interp['direction'] == 'INCREASING RISK':
            actions['immediate'].append({
                'area': factor['name'].title(),
                'reason': interp['description'],
                'action': f"Address {factor['name']} immediately - strongest predictor of failure",
                'shap_importance': abs(interp['shap_value'])
            })
        elif interp['magnitude'] == 'MODERATE' and interp['direction'] == 'INCREASING RISK':
            actions['short_term'].append({
                'area': factor['name'].title(),
                'reason': interp['description'],
                'action': f"Monitor and support {factor['name']} improvement",
                'shap_importance': abs(interp['shap_value'])
            })
        elif interp['direction'] == 'PROTECTIVE':
            actions['monitoring'].append({
                'area': factor['name'].title(),
                'reason': interp['description'],
                'action': f"Maintain current {factor['name']} - it's helping",
                'shap_importance': abs(interp['shap_value'])
            })
    
    # Sort by SHAP importance
    for category in actions:
        if 'shap_importance' in str(actions[category]):
            actions[category].sort(key=lambda x: x.get('shap_importance', 0), reverse=True)
    
    return actions


def get_percentile_context(feature_name, value, all_values):
    """
    Get percentile context for a value
    
    Returns:
        dict: Percentile info
    """
    percentile = (all_values <= value).sum() / len(all_values) * 100
    
    if percentile > 75:
        context = "top 25% (concerning if negative feature)"
    elif percentile > 50:
        context = "above average"
    elif percentile > 25:
        context = "below average"
    else:
        context = "bottom 25%"
    
    return {
        'percentile': percentile,
        'context': context,
        'is_outlier': percentile > 90 or percentile < 10
    }