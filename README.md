# Market Tool Stability Analysis

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
Results in stability_results directory
```

## Run Geographic Strategy
```bash
# Run analysis with geographic strategy
python run_analysis.py --strategy geographic --max-workers 12

# View results
Results in stability_results directory
```

## Run Benchmark City Strategy
```bash
# Run analysis with benchmark strategy
python run_analysis.py --strategy benchmark --max-workers 12

# View results
Results in stability_results directory
```

Each strategy will:
1. Load city data from data/cities.csv
2. Analyze cities from data/test_cities.csv
3. Output results to analysis_results/[strategy][time]_results.csv
