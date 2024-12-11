# Market Analysis Strategy Details

## Cluster Strategy
The cluster strategy uses KNN (K-Nearest Neighbors) to find similar cities based on demographics, then scores opportunities using both similarity and cluster-based performance metrics.

### Process:
1. **Similar City Selection**: 
   - First filters cities within specified radius (default 100 miles) of the target city
   - Uses KNN to find 15 most similar cities based on demographic features:
     - Population metrics (total, proper, density)
     - Age demographics (median, over 65)
     - Economic indicators (income, dual income families)
     - Housing metrics (ownership, units, value, rent)
     - Education levels
   - All similarity comparisons are made directly to the target city

2. **Scoring Components**:
   - Demographic Similarity (30%):
     - Normalized KNN distance scores from target city
     - Higher weight for more similar cities
   - Performance vs. Cluster (20%):
     - GA4 metrics comparison to cluster averages
     - Users, conversion rates, leads
   - Network Penetration (10%):
     - Network penetration vs. cluster average
   - Engagement Diversity (10%):
     - Unique sites relative to total users
   - Growth Potential (10%):
     - Room for growth vs. cluster average
   - Performance Efficiency (10%):
     - Lead generation relative to network size
   - Saturation Risk (10%):
     - Market saturation assessment

3. **Final Score Adjustment**:
   - Applies market size multiplier based on housing units
   - Normalizes to 0-1 range
   - Categorizes into Low/Average/High opportunity

## Benchmark City Strategy
This strategy uses a combination of similarity scores and cluster-based metrics, with all components matching the original engine implementation.

### Process:
1. **City Selection**:
   - First identifies highest performing city within 250 miles of target city as the benchmark
   - Then finds similar cities within 250 miles of the benchmark city (not target city)
   - Uses KNN for similarity scoring relative to the benchmark city
   - Uses cluster averages for performance metrics
   - All similarity comparisons are made to the benchmark city, not target city

2. **Scoring Components**:
   - Demographic Similarity (30%):
     - Normalized similarity scores from KNN relative to benchmark city
   - Performance vs. Cluster (20%):
     - GA4 metrics vs. cluster averages
   - Network Penetration (10%):
     - Network penetration vs. cluster average
   - Engagement Diversity (10%):
     - Unique sites relative to total users
   - Growth Potential (10%):
     - Room for growth vs. cluster average
   - Performance Efficiency (10%):
     - Lead generation relative to network size
   - Saturation Risk (10%):
     - Market saturation assessment

3. **Final Score Adjustment**:
   - Applies market size multiplier based on housing units
   - Normalizes to 0-1 range
   - Categorizes into Low/Average/High opportunity

## Geographic Strategy
This strategy takes a pure geographic approach, using all cities within a broad radius to establish regional context and benchmarks.

### Process:
1. **Regional Context**:
   - First gets all cities within 250 miles of target city
   - Uses this full set to calculate regional averages and benchmarks
   - No artificial limits on number of cities
   - All metrics are compared to regional averages, not nationwide clusters

2. **City Selection**:
   - Gets all cities within user-specified radius (e.g., 100 miles)
   - Calculates similarity scores for every city in range
   - No KNN filtering - keeps all cities for comparison
   - Direct demographic similarity calculation to target city

3. **Scoring Components**:
   - Demographic Similarity (30%):
     - Direct Euclidean distance on normalized demographic features
     - Higher weight for more similar cities
   - Performance vs. Region (20%):
     - GA4 metrics comparison to regional averages (250-mile radius)
     - Users, conversion rates, leads
   - Network Penetration (10%):
     - Network penetration vs. regional average
   - Engagement Diversity (10%):
     - Unique sites relative to total users
   - Growth Potential (10%):
     - Room for growth vs. regional average
   - Performance Efficiency (10%):
     - Lead generation relative to network size
   - Saturation Risk (10%):
     - Market saturation assessment

4. **Final Score Adjustment**:
   - Applies market size multiplier based on housing units
   - Normalizes to 0-1 range
   - Categorizes into Low/Average/High opportunity

## Key Differences
1. **Geographic Scope**:
   - Cluster Strategy:
     - Uses 100-mile radius from target city
     - Compares to nationwide clusters
     - Limits to 15 most similar cities
   - Benchmark Strategy:
     - Uses 250-mile radius for benchmark selection
     - Compares to nationwide clusters
     - Similarity based on benchmark city
   - Geographic Strategy:
     - Uses 250-mile radius for regional context
     - Compares only to regional averages
     - No limit on number of similar cities

2. **Comparison Basis**:
   - Cluster Strategy: Nationwide cluster averages
   - Benchmark Strategy: Highest performing nearby city
   - Geographic Strategy: Regional averages (250-mile radius)

3. **Similarity Calculation**:
   - Cluster Strategy: KNN to target city
   - Benchmark Strategy: KNN to benchmark city
   - Geographic Strategy: Direct distance to target city

4. **Common Elements**:
   - All use standardized GA4 metrics
   - All apply same market size adjustment
   - All use same scoring component weights
   - All use same risk assessment approach
