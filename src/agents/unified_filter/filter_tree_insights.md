# Unified Filter Tree Structure Insights

Based on the test results, here's the structure we should use for our models:

## Key Learnings

### 1. **Global vs Location-Specific Filters**
The LLM correctly identifies:
- **Global filters**: Apply to all locations (e.g., date ranges, common demographics)
- **Location-specific filters**: Apply only within a location context

### 2. **Location Context Structure**
Each location has:
```json
{
  "location_name": "Dubai",
  "location_index": 0,  // For joining with query_location_geohashes
  "radius_meters": null,  // Can be specified if needed
  "time_filters": {
    "date_ranges": [],
    "hour_ranges": [],
    "day_of_week": [],
    "is_weekend": true/false/null
  },
  "profile_filters": {
    "filter_tree": {...}
  },
  "risk_filters": {
    "filter_tree": {...}
  }
}
```

### 3. **Time Filter Flexibility**
Time filters can be:
- Global (apply to all locations)
- Location-specific (different times for different locations)
- Include various constraints:
  - Date ranges
  - Hour ranges (24-hour format)
  - Day of week (1=Monday, 7=Sunday)
  - Weekend/weekday flags

### 4. **Query Optimization Metadata**
The LLM provides:
- `suggested_data_source`: "movement_data", "profile_only", "aggregated_location"
- `data_access_pattern`: "location_time_series", "profile_lookup", "co_location"
- `optimization_hints`: Specific strategies for the query
- `estimated_selectivity`: "high", "medium", "low"

### 5. **Filter Complexity Indicators**
The metadata includes:
- `has_location_specific_filters`: Boolean
- `requires_location_join`: Boolean
- `location_count`: Number of locations
- `has_time_varying_filters`: Boolean
- `filter_complexity`: "high", "medium", "low"

## Recommended Model Structure

Based on these insights, our models should:

1. **Keep the LocationContext class** as designed - it matches the LLM output perfectly
2. **UnifiedFilterTree class** should have:
   - `global_filters` (with filter_tree and exclusions)
   - `location_contexts` (list of LocationContext)
   - `query_optimization` (dict with suggestions)
   - `metadata` (dict with complexity indicators)

3. **No need for separate time_ranges** at the top level - they're part of either global_filters or location_contexts

## SQL Generation Strategy

With this structure, we can:

1. **Profile-only queries**: When no location_contexts exist
2. **Single location queries**: One location_context with filters
3. **Multi-location queries**: Multiple location_contexts, possibly requiring UNION
4. **Pattern queries**: Same users in different locations at different times

## Next Steps

1. Finalize the model structure based on LLM output patterns
2. Create the SQL Generator agent that:
   - Takes orchestrator output
   - Calls LLM to create unified filter tree
   - Translates filter tree to SQL (or other query languages)
3. Test with real orchestrator outputs