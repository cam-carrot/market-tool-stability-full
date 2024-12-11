import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from typing import Dict, Optional
import logging
from .base_strategy import MarketStrategy

class BenchmarkCityStrategy(MarketStrategy):
    """Strategy that uses the highest performing city within 250 miles as a benchmark"""
    
    def __init__(self, city_data: pd.DataFrame):
        """Initialize the benchmark city strategy"""
        super().__init__(city_data)
        self.logger = logging.getLogger(__name__)
        self.benchmark_cache: Dict[str, str] = {}  # Cache benchmark cities
        self._calculate_cluster_averages()

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

    def get_strategy_name(self) -> str:
        return "Benchmark City Analysis"

    def get_strategy_description(self) -> str:
        return ("Analyzes markets by comparing to the highest performing city within "
                "250 miles. This provides a real-world benchmark for performance "
                "potential in the geographic region.")

    def find_similar_cities(self, target_city: str, target_state: str, radius_miles: int) -> pd.DataFrame:
        """Find similar cities using the benchmark city as a reference point"""
        target_city_state = f"{target_city}, {target_state}".lower()
        
        # Get or find the benchmark city
        benchmark_city = self._get_benchmark_city(target_city_state)
        
        # Get cities within the test radius
        nearby_cities = self._filter_by_distance(target_city_state, radius_miles)
        
        # Add benchmark city if not in radius
        if benchmark_city not in nearby_cities.index:
            self.logger.info(f"Benchmark city {benchmark_city} not in current radius, adding it")
            benchmark_city_data = self.city_data.loc[[benchmark_city]]
            nearby_cities = pd.concat([nearby_cities, benchmark_city_data])
        
        # Define demographic features for similarity comparison
        features = [
            'population', 'population_proper', 'density', 'incorporated', 'age_median',
            'age_over_65', 'family_dual_income', 'income_household_median', 'income_household_six_figure',
            'home_ownership', 'housing_units', 'home_value', 'rent_median', 'education_college_or_above',
            'race_white', 'race_black', 'hispanic', 'income_individual_median', 'rent_burden', 'poverty'
        ]
        
        # Clean and normalize data
        nearby_cities = self.clean_data(nearby_cities, features)
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(nearby_cities[features])
        
        # Use equal weights for all features
        weights = np.ones(len(features))
        weighted_data = normalized_data * weights.reshape(1, -1)
        
        # Find similar cities using KNN
        n_neighbors = min(14, len(nearby_cities) - 1)  # -1 to leave room for target city
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        nn.fit(weighted_data)
        
        # Use benchmark city as center point for KNN
        center_index = nearby_cities.index.get_loc(benchmark_city)
        distances, indices = nn.kneighbors(weighted_data[center_index].reshape(1, -1))
        
        # Get similar cities
        similar_cities_indices = nearby_cities.index[indices[0]]
        similar_cities = nearby_cities.loc[similar_cities_indices].copy()
        
        # Add target city if not included
        if target_city_state not in similar_cities.index:
            target_city_data = nearby_cities.loc[[target_city_state]]
            similar_cities = pd.concat([similar_cities, target_city_data])
        
        # Calculate distances and similarity scores
        similar_cities['distance_to_target'] = self.haversine_distances(
            similar_cities[['lat', 'lng']].values,
            self.city_data.loc[target_city_state, ['lat', 'lng']].values.flatten()
        )
        
        # Set similarity scores
        similar_cities['similarity_score'] = np.inf  # Default value
        similar_cities.loc[similar_cities_indices, 'similarity_score'] = distances[0]
        similar_cities.loc[target_city_state, 'similarity_score'] = 0  # Target city gets perfect similarity
        
        # Calculate opportunity scores
        similar_cities = self.calculate_opportunity_score(similar_cities, target_city_state)
        similar_cities = similar_cities.sort_values('opportunity_score', ascending=False)
        
        return similar_cities

    def calculate_opportunity_score(self, df: pd.DataFrame, target_city_state: str) -> pd.DataFrame:
        """Calculate opportunity scores using cluster-based comparisons."""
        self.logger.info(f"Calculating opportunity score. DataFrame shape: {df.shape}")
        
        # GA4 metrics standardization
        ga4_columns = ['users_org', 'cvr_org', 'leads_org', 'users_paid', 'cvr_paid', 'leads_paid']
        scaler = StandardScaler()
        std_ga4_columns = [f'std_{col}' for col in ga4_columns]
        
        # Calculate performance difference based on cluster averages
        df['performance_diff'] = df.apply(
            lambda row: self._calculate_cluster_performance_diff(row, ga4_columns), 
            axis=1
        )

        # Calculate network penetration relative to cluster average
        df['network_penetration'] = df.apply(
            lambda row: (row['unique_sites'] / row['housing_units'] * 100) / 
                       self.cluster_averages[row['cluster']]['network_penetration'],
            axis=1
        )

        # Calculate other metrics
        df['engagement_diversity'] = df['unique_sites'] / (df['users_org'] + df['users_paid'] + 1)
        
        # Growth potential based on cluster average
        df['growth_potential'] = df.apply(
            lambda row: (self.cluster_averages[row['cluster']]['network_penetration'] - 
                        (row['unique_sites'] / row['housing_units'] * 100)) / 
                        self.cluster_averages[row['cluster']]['network_penetration'],
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

    def _calculate_cluster_performance_diff(self, row: pd.Series, metrics: list) -> float:
        """Calculate performance difference based on cluster averages"""
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

    def _get_benchmark_city(self, target_city_state: str) -> str:
        """Get or find the benchmark city for the target"""
        if target_city_state not in self.benchmark_cache:
            # Find highest performing city within 250 miles
            max_radius_cities = self._filter_by_distance(target_city_state, 250)
            max_radius_cities['total_leads'] = max_radius_cities['leads_org'] + max_radius_cities['leads_paid']
            benchmark_city = max_radius_cities['total_leads'].idxmax()
            self.benchmark_cache[target_city_state] = benchmark_city
        
        return self.benchmark_cache[target_city_state]

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

    def clean_data(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """Clean and impute missing data."""
        for feature in features:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
        imputer = SimpleImputer(strategy='median')
        df[features] = imputer.fit_transform(df[features])
        return df
