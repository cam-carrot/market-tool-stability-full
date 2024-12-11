import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging
from .base_strategy import MarketStrategy

class GeographicStrategy(MarketStrategy):
    """Strategy that analyzes markets based on geographic proximity and local region metrics"""
    
    def __init__(self, city_data: pd.DataFrame):
        """Initialize the geographic strategy"""
        super().__init__(city_data)
        self.logger = logging.getLogger(__name__)
        self.region_averages = {}  # Cache for region averages

    def get_strategy_name(self) -> str:
        return "Geographic Region Analysis"

    def get_strategy_description(self) -> str:
        return ("Analyzes markets by comparing to all cities within 250 miles to establish "
                "regional benchmarks, then scores opportunities based on local market dynamics.")

    def find_similar_cities(self, target_city: str, target_state: str, radius_miles: int) -> pd.DataFrame:
        """Find and score cities using geographic proximity and regional metrics"""
        target_city_state = f"{target_city}, {target_state}".lower()
        
        # First get all cities within 250 miles for regional context and scoring
        region_cities = self._filter_by_distance(target_city_state, 250)
        self.logger.info(f"Cities within 250 miles for regional context: {len(region_cities)}")
        
        # Calculate regional averages for GA4 metrics
        self._calculate_region_averages(region_cities)
        
        # Clean and normalize data for similarity calculation
        features = [
            'population', 'population_proper', 'density', 'incorporated', 'age_median',
            'age_over_65', 'family_dual_income', 'income_household_median', 'income_household_six_figure',
            'home_ownership', 'housing_units', 'home_value', 'rent_median', 'education_college_or_above',
            'race_white', 'race_black', 'hispanic', 'income_individual_median', 'rent_burden', 'poverty'
        ]
        
        # Clean and normalize all cities in 250 mile radius
        region_cities = self.clean_data(region_cities, features)
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(region_cities[features])
        
        # Calculate similarity scores for all cities in region
        target_normalized = normalized_data[region_cities.index.get_loc(target_city_state)]
        similarity_distances = np.sqrt(np.sum((normalized_data - target_normalized) ** 2, axis=1))
        
        # Calculate distances and set similarity scores for all regional cities
        region_cities['distance_to_target'] = self.haversine_distances(
            region_cities[['lat', 'lng']].values,
            self.city_data.loc[target_city_state, ['lat', 'lng']].values.flatten()
        )
        region_cities['similarity_score'] = similarity_distances
        region_cities.loc[target_city_state, 'similarity_score'] = 0  # Target city gets perfect similarity
        
        # Calculate opportunity scores for all cities in region
        region_cities = self.calculate_opportunity_score(region_cities, target_city_state)
        
        # Now filter down to just cities within requested radius
        if radius_miles != 250:
            similar_cities = region_cities[region_cities['distance_to_target'] <= radius_miles].copy()
            self.logger.info(f"Filtered to {len(similar_cities)} cities within {radius_miles} miles")
        else:
            similar_cities = region_cities.copy()
        
        # Sort by opportunity score
        similar_cities = similar_cities.sort_values('opportunity_score', ascending=False)
        
        return similar_cities

    def calculate_opportunity_score(self, df: pd.DataFrame, target_city_state: str) -> pd.DataFrame:
        """Calculate opportunity scores using regional comparisons."""
        self.logger.info(f"Calculating opportunity score. DataFrame shape: {df.shape}")
        
        # GA4 metrics standardization
        ga4_columns = ['users_org', 'cvr_org', 'leads_org', 'users_paid', 'cvr_paid', 'leads_paid']
        
        # Calculate performance difference based on regional averages
        df['performance_diff'] = df.apply(
            lambda row: self._calculate_performance_diff(row, ga4_columns), 
            axis=1
        )

        # Calculate network penetration relative to regional average
        df['network_penetration'] = df.apply(
            lambda row: (row['unique_sites'] / row['housing_units'] * 100) / 
                       self.region_averages['network_penetration'],
            axis=1
        )

        # Calculate other metrics
        df['engagement_diversity'] = df['unique_sites'] / (df['users_org'] + df['users_paid'] + 1)
        
        # Growth potential based on regional average
        df['growth_potential'] = df.apply(
            lambda row: (self.region_averages['network_penetration'] - 
                        (row['unique_sites'] / row['housing_units'] * 100)) / 
                        self.region_averages['network_penetration'],
            axis=1
        )
        
        df['performance_efficiency'] = (df['leads_org'] + df['leads_paid']) / (df['unique_sites'] + 1)
        
        # Saturation risk calculation
        df['log_unique_sites'] = np.log1p(df['unique_sites'])
        df['log_housing_units'] = np.log1p(df['housing_units'])
        df['saturation_risk'] = 1 - (1 / (1 + np.exp(-(df['log_unique_sites'] - df['log_housing_units']))))

        # Normalize similarity score
        if df['similarity_score'].isna().all():
            df['norm_similarity'] = 1
        else:
            df['norm_similarity'] = 1 - (
                (df['similarity_score'] - df['similarity_score'].min()) /
                (df['similarity_score'].max() - df['similarity_score'].min())
            )

        # Calculate raw opportunity score
        df['raw_opportunity_score'] = (
            0.3 * df['norm_similarity'] +
            0.2 * (1 - df['performance_diff']) +
            0.1 * df['network_penetration'] +
            0.1 * df['engagement_diversity'] +
            0.1 * df['growth_potential'] +
            0.1 * df['performance_efficiency'] +
            0.1 * (1 - df['saturation_risk'])
        )

        # Apply market size adjustment
        df['normalized_log_housing'] = (df['log_housing_units'] - df['log_housing_units'].min()) / (
            df['log_housing_units'].max() - df['log_housing_units'].min()
        )
        df['opportunity_score'] = df['raw_opportunity_score'] * (1 + df['normalized_log_housing'])

        # Normalize final score to 0-1 range
        min_score = df['opportunity_score'].min()
        max_score = df['opportunity_score'].max()
        if min_score != max_score:
            df['opportunity_score'] = (df['opportunity_score'] - min_score) / (max_score - min_score)
        else:
            df['opportunity_score'] = 1

        # Categorize scores
        df['opportunity_category'] = pd.qcut(
            df['opportunity_score'],
            q=3,
            labels=['Low', 'Average', 'High'],
            duplicates='drop'
        )

        return df

    def _calculate_region_averages(self, region_cities: pd.DataFrame):
        """Calculate and store average metrics for the region"""
        ga4_columns = ['users_org', 'cvr_org', 'leads_org', 'users_paid', 'cvr_paid', 'leads_paid', 'unique_sites']
        
        self.region_averages = {
            'ga4_metrics': {col: region_cities[col].mean() for col in ga4_columns},
            'network_penetration': (region_cities['unique_sites'] / region_cities['housing_units']).mean() * 100
        }

    def _calculate_performance_diff(self, row: pd.Series, metrics: list) -> float:
        """Calculate performance difference based on regional averages"""
        region_avgs = self.region_averages['ga4_metrics']
        
        # Calculate normalized differences for each metric
        diffs = []
        for col in metrics:
            if region_avgs[col] != 0:
                diff = (row[col] - region_avgs[col]) / region_avgs[col]
            else:
                diff = 0
            diffs.append(diff)
            
        # Return average difference
        return np.mean(diffs)

    def _filter_by_distance(self, target_city_state: str, radius_miles: int) -> pd.DataFrame:
        """Filter cities within specified radius"""
        target_lat, target_lng = self.city_data.loc[target_city_state, ['lat', 'lng']]
        distances = self.haversine_distances(
            self.city_data[['lat', 'lng']].values,
            np.array([target_lat, target_lng])
        )
        nearby_mask = distances <= radius_miles
        return self.city_data[nearby_mask].copy()

    @staticmethod
    def haversine_distances(points, target):
        """Calculate haversine distances between points."""
        R = 3959.87433  # Earth's radius in miles
        if points.ndim == 1:
            points = points.reshape(1, -1)
        lat1, lon1 = np.radians(points[:, 0]), np.radians(points[:, 1])
        lat2, lon2 = np.radians(target[0]), np.radians(target[1])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c
