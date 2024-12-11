from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from sklearn.impute import SimpleImputer

@dataclass
class CityAnalysisResult:
    city_state: str
    scores: Dict[int, float]
    categories: Dict[int, str]
    mean_score: float
    std_dev: float
    cv: float
    category_changes: int
    min_score: float
    max_score: float
    score_range: float
    is_zero_score: bool
    performance_metrics: Optional[Dict[int, Dict[str, float]]] = None

class MarketStrategy(ABC):
    """Abstract base class for market analysis strategies"""
    
    def __init__(self, city_data: pd.DataFrame):
        # Convert lat/lng to numeric before setting index
        city_data['lat'] = pd.to_numeric(city_data['lat'], errors='coerce')
        city_data['lng'] = pd.to_numeric(city_data['lng'], errors='coerce')
        self.city_data = city_data
    
    @abstractmethod
    def find_similar_cities(self, target_city: str, target_state: str, radius_miles: int) -> pd.DataFrame:
        """Find similar cities based on the strategy's criteria"""
        pass

    @abstractmethod
    def calculate_opportunity_score(self, df: pd.DataFrame, target_city_state: str) -> pd.DataFrame:
        """Calculate opportunity scores based on the strategy's methodology"""
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of the strategy"""
        pass

    @abstractmethod
    def get_strategy_description(self) -> str:
        """Return a description of how the strategy works"""
        pass

    def clean_data(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Clean and prepare data for analysis"""
        # Create imputer for missing values
        imputer = SimpleImputer(strategy='median')
        
        # Impute missing values
        df_cleaned = df.copy()
        df_cleaned[features] = imputer.fit_transform(df[features])
        
        # Convert percentage strings to floats
        for col in features:
            if df_cleaned[col].dtype == 'object':
                df_cleaned[col] = df_cleaned[col].str.rstrip('%').astype('float') / 100.0
                
        return df_cleaned
