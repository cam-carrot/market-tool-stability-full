import pandas as pd
import numpy as np
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Type
import matplotlib.pyplot as plt
import seaborn as sns

from market_strategies.base_strategy import MarketStrategy, CityAnalysisResult
from market_strategies.cluster_strategy import ClusterStrategy
from market_strategies.benchmark_city_strategy import BenchmarkCityStrategy

class StabilityAnalyzer:
    """Main stability analysis class that can use different market analysis strategies"""
    
    def __init__(self, 
                 city_data_path: str,
                 strategy_class: Type[MarketStrategy],
                 test_cities_path: str,
                 output_dir: str = "stability_results"):
        """
        Initialize the analyzer with a specific strategy
        
        Args:
            city_data_path: Path to the city data CSV file
            strategy_class: Class of the strategy to use (e.g., ClusterStrategy)
            test_cities_path: Path to the test cities CSV file
            output_dir: Directory to save analysis results
        """
        self.logger = logging.getLogger(__name__)
        
        # Create timestamped output directory with strategy name
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        strategy_name = strategy_class.__name__.replace('Strategy', '').lower()
        self.output_dir = Path(output_dir) / f"{strategy_name}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load city data
        self.city_data = pd.read_csv(city_data_path)
        self._standardize_city_names()
        
        # Initialize strategy
        self.strategy = strategy_class(self.city_data)
        self.logger.info(f"Using strategy: {self.strategy.get_strategy_name()}")

    def analyze_city(self, city_state: str, radius_list: List[int]) -> Optional[CityAnalysisResult]:
        """Analyze a single city across multiple radii"""
        try:
            # Clean and standardize the city_state string
            city_state = city_state.strip().strip('\"').strip("'")
            city, state = city_state.rsplit(',', 1)
            city = city.strip()
            state = state.strip()
            
            self.logger.info(f"Analyzing {city}, {state}")
            
            scores = {}
            categories = {}
            performance_metrics = {}
            
            # Create standardized lookup key
            lookup_key = f"{city}, {state}".lower()
            
            # Check if city exists in dataset
            if lookup_key not in self.city_data.index:
                raise ValueError(f"City '{city}, {state}' not found in dataset")
            
            for radius in radius_list:
                similar_cities = self.strategy.find_similar_cities(city, state, radius)
                
                # Calculate scores using the strategy
                scored_cities = self.strategy.calculate_opportunity_score(similar_cities, lookup_key)
                if lookup_key not in scored_cities.index:
                    raise ValueError(f"City '{city}, {state}' not found in scored cities")
                
                city_data = scored_cities.loc[lookup_key]
                scores[radius] = city_data['opportunity_score']
                categories[radius] = city_data['opportunity_category']
                
                # Store performance metrics
                performance_metrics[radius] = {
                    'performance_diff': city_data['performance_diff'],
                    'network_penetration': city_data['network_penetration'],
                    'growth_potential': city_data['growth_potential'],
                    'performance_efficiency': city_data['performance_efficiency'],
                    'saturation_risk': city_data['saturation_risk']
                }
            
            score_values = list(scores.values())
            is_zero_score = all(score == 0 for score in score_values)
            
            # Calculate coefficient of variation
            cv = 0.0 if is_zero_score else (
                np.std(score_values) / np.mean(score_values) 
                if np.mean(score_values) != 0 else np.inf
            )
            
            # Count category changes
            category_values = list(categories.values())
            category_changes = sum(1 for i in range(len(category_values)-1)
                                 if category_values[i] != category_values[i+1])
            
            return CityAnalysisResult(
                city_state=f"{city}, {state}",
                scores=scores,
                categories=categories,
                mean_score=np.mean(score_values),
                std_dev=np.std(score_values),
                cv=cv,
                category_changes=category_changes,
                min_score=min(score_values),
                max_score=max(score_values),
                score_range=max(score_values) - min(score_values),
                is_zero_score=is_zero_score,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing {city_state}: {str(e)}")
            return None

    def run_analysis(self, test_cities: List[str], radius_list: List[int], max_workers: int = 6) -> List[CityAnalysisResult]:
        """Run stability analysis on multiple cities"""
        self.logger.info(f"Starting analysis of {len(test_cities)} cities across {len(radius_list)} radii")
        
        # Split cities into chunks for parallel processing
        chunk_size = max(1, len(test_cities) // (max_workers * 2))  # Ensure enough tasks for all workers
        city_chunks = [test_cities[i:i + chunk_size] for i in range(0, len(test_cities), chunk_size)]
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit chunks of cities for parallel processing
            future_to_chunk = {
                executor.submit(self._analyze_city_chunk, chunk, radius_list): i 
                for i, chunk in enumerate(city_chunks)
            }
            
            # Process results as they complete
            for future in future_to_chunk:
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    self.logger.info(f"Completed chunk {future_to_chunk[future]} ({len(chunk_results)} cities)")
                except Exception as e:
                    self.logger.error(f"Error in chunk {future_to_chunk[future]}: {str(e)}")
        
        self._generate_outputs(results, radius_list)
        return results

    def _analyze_city_chunk(self, cities: List[str], radius_list: List[int]) -> List[CityAnalysisResult]:
        """Analyze a chunk of cities"""
        chunk_results = []
        for city_state in cities:
            try:
                result = self.analyze_city(city_state, radius_list)
                if result:
                    chunk_results.append(result)
            except Exception as e:
                self.logger.error(f"Error analyzing {city_state}: {str(e)}")
        return chunk_results

    def _standardize_city_names(self):
        """Standardize city names in the dataset"""
        # Use the existing city_state column
        if 'city_state' in self.city_data.columns:
            self.city_data['city_state'] = self.city_data['city_state'].str.strip().str.lower()
            self.city_data.set_index('city_state', inplace=True)
            self.logger.info(f"Standardized {len(self.city_data)} city names")

    def _generate_outputs(self, results: List[CityAnalysisResult], radius_list: List[int]):
        """Generate analysis outputs and visualizations"""
        if not results:
            self.logger.error("No results to generate outputs from")
            return

        # Get timestamp and strategy name for file names
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        strategy_name = self.strategy.__class__.__name__.replace('Strategy', '').lower()
        
        # Create results DataFrame
        df = self._create_results_df(results)
        df.to_csv(self.output_dir / f'stability_results_{strategy_name}_{timestamp}.csv', index=False)
        
        # Generate visualizations
        if len(results) > 0:
            self._plot_score_variations(results, radius_list)
            self._plot_stability_metrics(results)
            self._generate_report(df)

    def _create_results_df(self, results: List[CityAnalysisResult]) -> pd.DataFrame:
        """Create a DataFrame with all stability metrics"""
        data = []
        for r in results:
            row = {
                'city_state': r.city_state,
                'mean_score': r.mean_score,
                'std_dev': r.std_dev,
                'coefficient_of_variation': r.cv,
                'category_changes': r.category_changes,
                'score_range': r.score_range,
                'min_score': r.min_score,
                'max_score': r.max_score,
                'is_zero_score': r.is_zero_score,
                'stability_rating': self._get_stability_rating(r.cv, r.is_zero_score)
            }
            
            # Add scores and categories for each radius
            for radius, score in r.scores.items():
                row[f'score_radius_{radius}'] = score
                row[f'category_radius_{radius}'] = r.categories[radius]
                
                # Add performance metrics for each radius
                if r.performance_metrics:
                    metrics = r.performance_metrics[radius]
                    for metric_name, value in metrics.items():
                        row[f'{metric_name}_radius_{radius}'] = value
                
            data.append(row)
        
        return pd.DataFrame(data)

    def _get_stability_rating(self, cv: float, is_zero_score: bool = False) -> str:
        """Rate stability based on coefficient of variation"""
        if is_zero_score:
            return "Excellent"
        
        if cv < 0.1:
            return "Excellent"
        elif cv < 0.2:
            return "Good"
        elif cv < 0.3:
            return "Fair"
        elif cv < 0.4:
            return "Poor"
        else:
            return "Unstable"

    def _plot_score_variations(self, results: List[CityAnalysisResult], radius_list: List[int]):
        """Plot score variations across radii"""
        plt.figure(figsize=(12, 6))
        
        for result in results:
            plt.plot(radius_list, [result.scores[r] for r in radius_list],
                    alpha=0.3, marker='o', label=result.city_state)
        
        plt.title('Opportunity Score Variation Across Radii')
        plt.xlabel('Radius (miles)')
        plt.ylabel('Opportunity Score')
        plt.grid(True, alpha=0.3)
        
        if len(results) <= 10:  # Only show legend if we have 10 or fewer cities
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        strategy_name = self.strategy.__class__.__name__.replace('Strategy', '').lower()
        plt.tight_layout()
        plt.savefig(self.output_dir / f'score_variations_{strategy_name}_{timestamp}.png', bbox_inches='tight')
        plt.close()

    def _plot_stability_metrics(self, results: List[CityAnalysisResult]):
        """Plot stability metrics distributions"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Coefficient of Variation distribution
        cvs = [r.cv for r in results]
        sns.histplot(cvs, ax=ax1)
        ax1.set_title('Distribution of Coefficient of Variation')
        ax1.set_xlabel('Coefficient of Variation')
        
        # Score Range vs Mean Score
        means = [r.mean_score for r in results]
        ranges = [r.score_range for r in results]
        ax2.scatter(means, ranges, alpha=0.5)
        ax2.set_title('Score Range vs Mean Score')
        ax2.set_xlabel('Mean Score')
        ax2.set_ylabel('Score Range')
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        strategy_name = self.strategy.__class__.__name__.replace('Strategy', '').lower()
        plt.tight_layout()
        plt.savefig(self.output_dir / f'stability_metrics_{strategy_name}_{timestamp}.png')
        plt.close()

    def _generate_report(self, df: pd.DataFrame):
        """Generate a summary report of the analysis"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        strategy_name = self.strategy.__class__.__name__.replace('Strategy', '').lower()
        report_path = self.output_dir / f'analysis_report_{strategy_name}_{timestamp}.txt'
        
        with open(report_path, 'w') as f:
            f.write(f"Market Stability Analysis Report\n")
            f.write(f"==============================\n\n")
            
            f.write(f"Analysis Strategy: {self.strategy.get_strategy_name()}\n")
            f.write(f"Strategy Description: {self.strategy.get_strategy_description()}\n\n")
            
            f.write("Summary Statistics:\n")
            f.write("-----------------\n")
            f.write(f"Total cities analyzed: {len(df)}\n")
            f.write(f"Average CV: {df['coefficient_of_variation'].mean():.3f}\n\n")
            
            f.write("Stability Ratings Distribution:\n")
            f.write(f"{df['stability_rating'].value_counts().to_string()}\n\n")
            
            f.write("Most Stable Cities:\n")
            f.write("-----------------\n")
            most_stable = df.nsmallest(5, 'coefficient_of_variation')[
                ['city_state', 'coefficient_of_variation', 'stability_rating']
            ]
            f.write(f"{most_stable.to_string()}\n\n")
            
            f.write("Least Stable Cities:\n")
            f.write("-----------------\n")
            least_stable = df.nlargest(5, 'coefficient_of_variation')[
                ['city_state', 'coefficient_of_variation', 'stability_rating']
            ]
            f.write(f"{least_stable.to_string()}\n")
