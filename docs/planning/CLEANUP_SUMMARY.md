# Duplicate Notebook Cleanup Summary

## Overview
Successfully cleaned up duplicate notebooks by matching based on content similarity (not file names).

## Results
- **Total duplicate groups found**: 107
- **Files kept**: 107 (one clean version per project)
- **Files removed**: 160 (duplicate copies)

## Strategy
The cleanup prioritized files in the following order:
1. **Files in `2025-11-05_ipynb_processed`** - These are the cleaned versions you've already processed
   - Within processed directory, preferred organized categories (01_Time_Series, 02_Finance, etc.) over:
     - `_Duplicates_To_Review` folder
     - `_Experiments` subdirectories  
     - `_Templates` subdirectories
2. **Files in organized subdirectories** (not root level)
3. **Files with better naming** (no "Copy" suffixes, no numbered suffixes)
4. **Files not in TODO or experiments folders**

## Breakdown
- **85 files kept** from processed directory (cleaned versions)
- **22 files kept** from other locations (no processed version available)
- **47 files removed** from processed directory (duplicates within processed dir)
- **113 files removed** from other locations (root level, TODO, experiments, etc.)

## Files Preserved
All kept files are documented in `cleanup_decisions.json` with reasons for each decision.

## Next Steps
- Review `cleanup_decisions.json` if you need to see specific decisions
- The remaining notebooks are now unique, well-organized copies
- Consider organizing any remaining files that aren't in the processed directory structure

