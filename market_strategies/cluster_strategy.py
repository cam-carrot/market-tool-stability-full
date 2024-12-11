import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from typing import Dict
from .base_strategy import MarketStrategy
from abc import ABC, abstractmethod

class ClusterStrategy(MarketStrategy):
    """Strategy that uses demographic clusters for market analysis"""
    
    def __init__(self, city_data: pd.DataFrame):
        super().__init__(city_data)
        self._calculate_cluster_averages()
    
    def find_similar_cities(self, target_city: str, target_state: str, radius_miles: int) -> pd.DataFrame:
        """Find similar cities using KNN based on demographics and geographic proximity"""
        target_city_state = f"{target_city}, {target_state}".lower()
        
        # First filter by distance to get local market
        nearby_cities = self.filter_by_distance(target_city_state, radius_miles)
        
        # Features for similarity comparison
        features = [
            'population', 'population_proper', 'density', 'incorporated', 'age_median',
            'age_over_65', 'family_dual_income', 'income_household_median', 'income_household_six_figure',
            'home_ownership', 'housing_units', 'home_value', 'rent_median', 'education_college_or_above',
            'race_white', 'race_black', 'hispanic', 'income_individual_median', 'rent_burden', 'poverty'
        ]
        
        # Clean and normalize features
        nearby_cities = self.clean_data(nearby_cities, features)
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(nearby_cities[features])
        
        # Set equal weights for all features
        feature_weights = {feature: 1 for feature in features}
        weights = np.array([feature_weights.get(feature, 1) for feature in features])
        weighted_data = normalized_data * weights.reshape(1, -1)
        
        # Find nearest neighbors
        n_neighbors = min(15, len(nearby_cities))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        nn.fit(weighted_data)
        
        target_idx = nearby_cities.index.get_loc(target_city_state)
        distances, indices = nn.kneighbors(weighted_data[target_idx].reshape(1, -1))
        
        # Get similar cities and add distance score
        similar_cities = nearby_cities.iloc[indices[0]].copy()
        
        # Calculate distance to target and similarity score
        similar_cities['distance_to_target'] = similar_cities.apply(
            lambda row: self.haversine_distances(np.array([row['lat'], row['lng']]), np.array([self.city_data.loc[target_city_state, 'lat'], self.city_data.loc[target_city_state, 'lng']])),
            axis=1
        )
        similar_cities['similarity_score'] = np.where(similar_cities.index == target_city_state, 0, distances[0])
        
        # Calculate opportunity scores using cluster-based comparisons
        similar_cities = self.calculate_opportunity_score(similar_cities, target_city_state)
        similar_cities = similar_cities.sort_values('opportunity_score', ascending=False)
        
        return similar_cities

    def calculate_opportunity_score(self, df: pd.DataFrame, target_city_state: str) -> pd.DataFrame:
        """Calculate opportunity scores using cluster-based comparisons"""
        # GA4 metrics standardization
        ga4_columns = ['users_org', 'cvr_org', 'leads_org', 'users_paid', 'cvr_paid', 'leads_paid']
        
        # Calculate performance difference based on cluster averages
        df['performance_diff'] = df.apply(
            lambda row: self._calculate_cluster_performance_diff(row, ga4_columns),
            axis=1
        )

        # Calculate network penetration relative to cluster average (using nationwide data)
        df['network_penetration'] = df.apply(
            lambda row: (row['unique_sites'] / row['housing_units'] * 100) /
                       self.cluster_averages[row['cluster']]['network_penetration'],
            axis=1
        )

        # Calculate growth potential using cluster average as target (using nationwide data)
        df['growth_potential'] = df.apply(
            lambda row: (self.cluster_averages[row['cluster']]['network_penetration'] -
                        (row['unique_sites'] / row['housing_units'] * 100)) /
                        self.cluster_averages[row['cluster']]['network_penetration'],
            axis=1
        )

        # Efficiency metrics
        df['performance_efficiency'] = (df['leads_org'] + df['leads_paid']) / (df['unique_sites'] + 1)
        df['engagement_diversity'] = df['unique_sites'] / (df['users_org'] + df['users_paid'] + 1)

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

        # Categorize scores into equal-sized bins (33% each)
        try:
            df['opportunity_category'] = pd.qcut(
                df['opportunity_score'],
                q=3,
                labels=['Low', 'Average', 'High']
            )
        except ValueError as e:
            # If we have too many duplicate values, use rank to break ties
            df['opportunity_score_rank'] = df['opportunity_score'].rank(method='first')
            df['opportunity_category'] = pd.qcut(
                df['opportunity_score_rank'],
                q=3,
                labels=['Low', 'Average', 'High']
            )
            df.drop('opportunity_score_rank', axis=1, inplace=True)

        return df

    def _calculate_cluster_performance_diff(self, row: pd.Series, metrics: list) -> float:
        """Calculate performance difference based on nationwide cluster averages"""
        cluster = row['cluster']
        cluster_avgs = self.cluster_averages[cluster]['ga4_metrics']
        
        # Calculate normalized differences for each metric
        diffs = []
        for col in metrics:
            if cluster_avgs[col] != 0:
                diff = (row[col] - cluster_avgs[col]) / cluster_avgs[col]
            else:
                diff = 0
            diffs.append(diff)
            
        # Return average difference
        return np.mean(diffs)

    def _calculate_cluster_averages(self):
        """Calculate and store average metrics for each cluster"""
        ga4_columns = ['users_org', 'cvr_org', 'leads_org', 'users_paid', 'cvr_paid', 'leads_paid', 'unique_sites']
        
        self.cluster_averages = {}
        for cluster in self.city_data['cluster'].unique():
            cluster_data = self.city_data[self.city_data['cluster'] == cluster]
            self.cluster_averages[cluster] = {
                'ga4_metrics': {col: cluster_data[col].mean() for col in ga4_columns},
                'network_penetration': (cluster_data['unique_sites'] / cluster_data['housing_units']).mean() * 100
            }

    def filter_by_distance(self, target_city_state: str, radius_miles: int) -> pd.DataFrame:
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

    def clean_data(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """Clean and prepare data for analysis."""
        # Convert features to numeric, forcing non-numeric to NaN
        for feature in features:
            if feature not in df.columns:
                df[feature] = np.nan
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
        
        # Convert lat/lng to numeric
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lng'] = pd.to_numeric(df['lng'], errors='coerce')
        
        # Drop rows where lat/lng are missing
        df = df.dropna(subset=['lat', 'lng'])
        
        # Fill remaining NaN values with median
        imputer = SimpleImputer(strategy='median')
        df[features] = imputer.fit_transform(df[features])
        return df

    def get_strategy_name(self) -> str:
        """Return the name of the strategy"""
        return "Cluster-Based Analysis"

    def get_strategy_description(self) -> str:
        """Return a description of how the strategy works"""
        return ("Analyzes markets by comparing cities within demographic clusters. "
                "Performance is measured relative to cluster averages, providing "
                "context-aware scoring that accounts for demographic similarities.")
