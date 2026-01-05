"""
explanations.py - Convert SHAP/LIME outputs to human language
NO hardcoded assumptions - fully generic for any dataset
"""

import numpy as np


def format_shap_explanation(feature_names, feature_values, shap_values, percentiles, all_data):
    """
    Convert SHAP values into human-readable explanations
    
    Args:
        feature_names: List of feature names
        feature_values: Student's actual feature values
        shap_values: SHAP values for the student
        percentiles: Percentile rank of each feature value
        all_data: DataFrame with all data for context
    
    Returns:
        list: List of explanation dictionaries
    """
    explanations = []
    
    # Combine and sort by absolute SHAP value (impact magnitude)
    feature_impacts = []
    for i, feature in enumerate(feature_names):
        feature_impacts.append({
            'feature': feature,
            'value': feature_values[feature],
            'shap_value': float(shap_values[i]),
            'abs_shap': abs(float(shap_values[i])),
            'percentile': percentiles[feature]
        })
    
    # Sort by impact (strongest first)
    feature_impacts.sort(key=lambda x: x['abs_shap'], reverse=True)
    
    # Generate explanations for each feature
    for item in feature_impacts:
        feature = item['feature']
        value = item['value']
        shap_val = item['shap_value']
        percentile = item['percentile']
        
        # Determine impact direction
        if shap_val > 0:
            impact_type = 'NEGATIVE'  # Increasing risk
            impact_word = 'increasing'
        else:
            impact_type = 'POSITIVE'  # Decreasing risk
            impact_word = 'reducing'
        
        # Determine magnitude
        if item['abs_shap'] > 0.15:
            magnitude = 'strongly'
        elif item['abs_shap'] > 0.08:
            magnitude = 'moderately'
        else:
            magnitude = 'slightly'
        
        # Determine position in dataset
        if percentile >= 90:
            position = 'in the top 10% (very high)'
        elif percentile >= 75:
            position = 'in the top 25% (high)'
        elif percentile >= 50:
            position = 'above average'
        elif percentile >= 25:
            position = 'below average'
        elif percentile >= 10:
            position = 'in the bottom 25% (low)'
        else:
            position = 'in the bottom 10% (very low)'
        
        # Calculate statistics for context
        median_val = all_data[feature].median()
        mean_val = all_data[feature].mean()
        
        # Build message
        message = f"Feature '{feature}' is {magnitude} {impact_word} risk"
        
        detail = (
            f"Current value: {value:.2f} ({position} compared to other students)\n"
            f"Class average: {mean_val:.2f} | Typical value: {median_val:.2f}\n"
            f"Impact strength: {item['abs_shap']:.3f}"
        )
        
        # Generate suggestion if negative impact
        if impact_type == 'NEGATIVE':
            if shap_val > 0 and value > median_val:
                # High value causing problems
                suggestion = (
                    f"Consider reducing this value toward the typical level ({median_val:.2f}). "
                    f"Even small improvements could help reduce risk."
                )
            elif shap_val > 0 and value <= median_val:
                # Low value but still causing problems
                suggestion = (
                    f"This feature is contributing to risk even though it's not unusually high. "
                    f"Focus on other stronger risk factors first."
                )
            else:
                suggestion = "Monitor this factor and work on improvement where possible."
        else:
            suggestion = None
        
        explanations.append({
            'feature': feature,
            'message': message,
            'detail': detail,
            'suggestion': suggestion,
            'impact_type': impact_type,
            'magnitude': magnitude,
            'shap_value': shap_val
        })
    
    return explanations


def format_lime_explanation(lime_explanation, feature_names):
    """
    Convert LIME rules into human-readable explanations
    
    Args:
        lime_explanation: LIME explanation object
        feature_names: List of feature names
    
    Returns:
        list: List of explanation dictionaries
    """
    explanations = []
    
    # Get LIME's feature contributions
    lime_list = lime_explanation.as_list()
    
    for rule, weight in lime_list:
        # Determine impact direction
        if weight > 0:
            impact_type = 'NEGATIVE'  # Increases risk
            direction_word = 'increases'
        else:
            impact_type = 'POSITIVE'  # Decreases risk
            direction_word = 'decreases'
        
        # Determine strength
        abs_weight = abs(weight)
        if abs_weight > 0.15:
            strength = 'strongly'
        elif abs_weight > 0.08:
            strength = 'moderately'
        else:
            strength = 'slightly'
        
        # Clean up the rule text
        rule_clean = rule.strip()
        
        # Build human-readable message
        message = f"When {rule_clean}, this {strength} {direction_word} the risk of poor performance"
        
        explanations.append({
            'rule': rule_clean,
            'message': message,
            'impact_type': impact_type,
            'strength': strength,
            'weight': weight
        })
    
    # Sort by absolute weight (strongest first)
    explanations.sort(key=lambda x: abs(x['weight']), reverse=True)
    
    return explanations


def get_risk_summary(risk_score, risk_category, confidence):
    """
    Generate a plain-language summary of the prediction
    
    Args:
        risk_score: Probability (0-1)
        risk_category: Category string
        confidence: Confidence level string
    
    Returns:
        str: Human-readable summary
    """
    if risk_category == 'High Risk':
        summary = (
            f"This student shows a high likelihood of struggling ({risk_score:.0%} risk score). "
            f"The model has {confidence.lower()} confidence in this prediction. "
            f"Immediate intervention is recommended."
        )
    elif risk_category == 'Medium Risk':
        summary = (
            f"This student shows moderate concerns ({risk_score:.0%} risk score). "
            f"The model has {confidence.lower()} confidence in this prediction. "
            f"Close monitoring and targeted support would be beneficial."
        )
    else:
        summary = (
            f"This student appears to be doing well ({risk_score:.0%} risk score). "
            f"The model has {confidence.lower()} confidence in this prediction. "
            f"Continue supporting current positive behaviors."
        )
    
    return summary


def suggest_interventions(shap_explanations, risk_level):
    """
    Generate actionable intervention suggestions based on explanations
    
    Args:
        shap_explanations: List of SHAP explanation dicts
        risk_level: Risk category string
    
    Returns:
        list: List of intervention suggestions
    """
    interventions = []
    
    # Focus on top negative factors
    negative_factors = [
        exp for exp in shap_explanations 
        if exp['impact_type'] == 'NEGATIVE'
    ][:3]  # Top 3
    
    for i, factor in enumerate(negative_factors, 1):
        if factor['suggestion']:
            interventions.append({
                'priority': i,
                'area': factor['feature'],
                'action': factor['suggestion']
            })
    
    # Add general recommendation based on risk level
    if risk_level == 'High Risk' and len(negative_factors) >= 2:
        interventions.append({
            'priority': 'Overall',
            'area': 'Comprehensive Support',
            'action': (
                'Multiple factors are contributing to risk. '
                'Consider a comprehensive support plan addressing the top areas simultaneously.'
            )
        })
    
    return interventions