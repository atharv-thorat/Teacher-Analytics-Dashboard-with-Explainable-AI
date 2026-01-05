"""
explanations.py - Format SHAP and LIME explanations
Ensures consistent output format for visualization
"""

import numpy as np


def format_shap_explanation(feature_names, feature_values, shap_values, percentiles, all_data):
    """
    Format SHAP values into a structured explanation
    
    Args:
        feature_names: List of feature names
        feature_values: Series/dict of feature values for the student
        shap_values: Array of SHAP values
        percentiles: Dict of percentile rankings
        all_data: DataFrame of all data for context
        
    Returns:
        List of dictionaries with formatted explanations
    """
    explanations = []
    
    # Convert to numpy array if needed
    if not isinstance(shap_values, np.ndarray):
        shap_values = np.array(shap_values)
    
    # Flatten if multi-dimensional
    if shap_values.ndim > 1:
        shap_values = shap_values.flatten()
    
    for i, feature in enumerate(feature_names):
        # Safely extract scalar values
        try:
            shap_val = float(shap_values[i])
        except (TypeError, ValueError):
            shap_val = float(np.array(shap_values[i]).item())
        
        try:
            feat_val = float(feature_values[feature])
        except (TypeError, ValueError, KeyError):
            feat_val = float(feature_values.iloc[i] if hasattr(feature_values, 'iloc') else feature_values[i])
        
        percentile = float(percentiles.get(feature, 50.0))
        
        # Calculate statistics for context
        try:
            mean_val = float(all_data[feature].mean())
            std_val = float(all_data[feature].std())
        except Exception:
            mean_val = 0.0
            std_val = 0.0
        
        # Determine if value is high or low relative to others
        if percentile < 25:
            relative_position = "very low"
        elif percentile < 50:
            relative_position = "below average"
        elif percentile < 75:
            relative_position = "above average"
        else:
            relative_position = "very high"
        
        explanations.append({
            "feature": feature,
            "feature_value": feat_val,
            "shap_value": shap_val,
            "percentile": percentile,
            "mean": mean_val,
            "std": std_val,
            "relative_position": relative_position,
            "impact_direction": "increases" if shap_val > 0 else "decreases"
        })
    
    # Sort by absolute SHAP value (most important first)
    explanations.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    
    return explanations


def format_lime_explanation(lime_explanation, feature_names):
    """
    Format LIME explanation into structured text
    
    Args:
        lime_explanation: LIME explanation object
        feature_names: List of feature names
        
    Returns:
        List of dictionaries with formatted text explanations
    """
    explanations = []
    
    # Get the explanation for the predicted class
    exp_list = lime_explanation.as_list()
    
    for feature_rule, weight in exp_list:
        # Determine if this increases or decreases risk
        direction = "increases" if weight > 0 else "decreases"
        
        # Format the message
        if weight > 0:
            message = f"✗ **{feature_rule}** increases risk (weight: {weight:.3f})"
            sentiment = "negative"
        else:
            message = f"✓ **{feature_rule}** reduces risk (weight: {abs(weight):.3f})"
            sentiment = "positive"
        
        explanations.append({
            "feature_rule": feature_rule,
            "weight": weight,
            "direction": direction,
            "sentiment": sentiment,
            "message": message
        })
    
    return explanations


def get_explanation_summary(shap_explanations, lime_explanations):
    """
    Generate a natural language summary of the explanations
    
    Args:
        shap_explanations: List of SHAP explanation dicts
        lime_explanations: List of LIME explanation dicts
        
    Returns:
        String with natural language summary
    """
    if not shap_explanations:
        return "No explanations available."
    
    # Get top 3 factors
    top_factors = shap_explanations[:3]
    
    # Build summary
    summary_parts = ["**Key Risk Factors:**\n"]
    
    for i, exp in enumerate(top_factors, 1):
        feature = exp["feature"]
        value = exp["feature_value"]
        shap_val = exp["shap_value"]
        percentile = exp["percentile"]
        relative = exp["relative_position"]
        
        if shap_val > 0:
            impact = "increases risk"
            emoji = "⬆️"
        else:
            impact = "decreases risk"
            emoji = "⬇️"
        
        summary_parts.append(
            f"{i}. {emoji} **{feature}** ({value:.1f}) is {relative} "
            f"(percentile: {percentile:.0f}%) and {impact}"
        )
    
    return "\n".join(summary_parts)


def format_feature_importance_text(importance_dict, top_n=5):
    """
    Format feature importance as readable text
    
    Args:
        importance_dict: Dictionary of feature importances
        top_n: Number of top features to include
        
    Returns:
        Formatted string
    """
    sorted_features = sorted(
        importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    lines = ["**Most Important Features for Risk Prediction:**\n"]
    for i, (feature, importance) in enumerate(sorted_features, 1):
        percentage = importance * 100
        lines.append(f"{i}. **{feature}**: {percentage:.1f}%")
    
    return "\n".join(lines)


def get_risk_interpretation(risk_score, risk_level, confidence):
    """
    Provide interpretation of risk score
    
    Args:
        risk_score: Probability score (0-1)
        risk_level: Category (Low/Medium/High Risk)
        confidence: Confidence level
        
    Returns:
        Dictionary with interpretation
    """
    interpretations = {
        "Low Risk": {
            "message": "Student is performing well and unlikely to need intervention",
            "emoji": "✅",
            "color": "success"
        },
        "Medium Risk": {
            "message": "Student shows some concerning indicators and may benefit from support",
            "emoji": "⚠️",
            "color": "warning"
        },
        "High Risk": {
            "message": "Student is at significant risk and requires immediate attention",
            "emoji": "🚨",
            "color": "error"
        }
    }
    
    base_interp = interpretations.get(risk_level, interpretations["Medium Risk"])
    
    return {
        **base_interp,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "confidence": confidence,
        "certainty_message": f"The model is {confidence.lower()} confidence in this prediction"
    }


def compare_students(student1_data, student2_data, feature_names):
    """
    Compare two students' features
    
    Args:
        student1_data: Dict/Series of student 1 features
        student2_data: Dict/Series of student 2 features  
        feature_names: List of features to compare
        
    Returns:
        Comparison summary
    """
    comparisons = []
    
    for feature in feature_names:
        val1 = student1_data[feature]
        val2 = student2_data[feature]
        diff = val1 - val2
        pct_diff = (diff / val2 * 100) if val2 != 0 else 0
        
        comparisons.append({
            "feature": feature,
            "student1": val1,
            "student2": val2,
            "difference": diff,
            "percent_difference": pct_diff
        })
    
    return comparisons