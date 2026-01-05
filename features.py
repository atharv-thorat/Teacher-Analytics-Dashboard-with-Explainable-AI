"""
features.py
Model-driven feature analysis - NO HARDCODED RULES
All thresholds and interpretations come from model outputs
"""

def risk_category(probability):
    """
    Converts risk probability to category using natural tertiles
    
    Args:
        probability: float between 0 and 1 (probability of failure)
    
    Returns:
        str: Risk category based on probability distribution
    """
    if probability < 0.33:
        return "Low Risk"
    elif probability < 0.67:
        return "Medium Risk"
    else:
        return "High Risk"


def confidence_score(probability):
    """
    Calculates confidence based on distance from decision boundary
    
    Args:
        probability: float between 0 and 1
    
    Returns:
        str: Confidence level
    """
    distance_from_boundary = abs(probability - 0.5)
    
    if distance_from_boundary > 0.35:
        return "Very High"
    elif distance_from_boundary > 0.20:
        return "High"
    elif distance_from_boundary > 0.10:
        return "Moderate"
    else:
        return "Low"