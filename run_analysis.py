import argparse
import logging
from pathlib import Path
from stability_analyzer import StabilityAnalyzer
from market_strategies.cluster_strategy import ClusterStrategy
from market_strategies.benchmark_city_strategy import BenchmarkCityStrategy
from market_strategies.geographic_strategy import GeographicStrategy
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_test_cities(test_cities_path: str) -> list:
    """Load and clean test cities from file"""
    test_cities = []
    with open(test_cities_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            if line.strip():
                city_state = line.strip().strip('\"').strip("'").lower()
                test_cities.append(city_state)
    return test_cities

def main():
    parser = argparse.ArgumentParser(description='Run market stability analysis')
    parser.add_argument('--strategy', type=str, choices=['cluster', 'benchmark', 'geographic'],
                      help='Analysis strategy to use', required=True)
    parser.add_argument('--data-dir', type=str, default='data',
                      help='Directory containing input data files')
    parser.add_argument('--output-dir', type=str, default='stability_results',
                      help='Directory for output files')
    parser.add_argument('--max-workers', type=int, default=6,
                      help='Maximum number of parallel workers')
    
    args = parser.parse_args()
    
    # Initialize paths
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            "Data directory not found. Please create a 'data' directory with your input files."
        )

    city_data_path = data_dir / "cities.csv"
    test_cities_path = data_dir / "test_cities.csv"

    # Verify input files exist
    for path, name in [(city_data_path, "cities.csv"),
                      (test_cities_path, "test_cities.csv")]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found at {path}")

    logger.info("Initializing analysis...")
    
    # Load data and initialize appropriate strategy
    city_data = pd.read_csv(city_data_path, low_memory=False)
    if args.strategy == 'cluster':
        strategy_class = ClusterStrategy
    elif args.strategy == 'benchmark':
        strategy_class = BenchmarkCityStrategy
    else:  # geographic
        strategy_class = GeographicStrategy
    
    # Initialize analyzer with selected strategy
    analyzer = StabilityAnalyzer(
        city_data_path=str(city_data_path),
        test_cities_path=str(test_cities_path),
        strategy_class=strategy_class,
        output_dir=args.output_dir
    )
    
    # Load test cities
    test_cities = load_test_cities(str(test_cities_path))
    logger.info(f"Loaded {len(test_cities)} test cities")
    
    # Define radius list
    radius_list = [50, 100, 150, 250]
    
    # Run stability analysis
    results = analyzer.run_analysis(test_cities, radius_list, max_workers=args.max_workers)
    
    if not results:
        logger.error("No results were generated. Please check your input data and city names.")
        return
        
    logger.info(f"Analysis complete. Results saved in '{args.output_dir}' directory")
    
    # Print summary from results file
    results_file = Path(args.output_dir) / 'stability_results.csv'
    if results_file.exists():
        df = pd.read_csv(results_file)
        print("\nSummary Statistics:")
        print("-" * 50)
        print(f"Total cities analyzed: {len(df)}")
        print(f"Average CV: {df['coefficient_of_variation'].mean():.3f}")
        print("\nStability Ratings Distribution:")
        print(df['stability_rating'].value_counts())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise
