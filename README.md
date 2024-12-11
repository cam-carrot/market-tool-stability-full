# Market Analysis Tool

Steps to run analysis with each strategy.

## Setup
```bash
# Install requirements
pip install -r requirements.txt
```

## Run Cluster Strategy
```bash
# Run analysis with cluster strategy
python run_analysis.py --strategy cluster --max-workers 12

# View results
cat analysis_results/cluster_results.csv
```

## Run Geographic Strategy
```bash
# Run analysis with geographic strategy
python run_analysis.py --strategy geographic --max-workers 12

# View results
cat analysis_results/geographic_results.csv
```

## Run Benchmark City Strategy
```bash
# Run analysis with benchmark strategy
python run_analysis.py --strategy benchmark --max-workers 12

# View results
cat analysis_results/benchmark_results.csv
```

Each strategy will:
1. Load city data from data/cities.csv
2. Analyze cities from data/test_cities.csv
3. Output results to analysis_results/[strategy]_results.csv
