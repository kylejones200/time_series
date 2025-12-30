# Repository Cleanup & Organization Plan

## Current State Analysis

### Statistics
- **26,104 Python files** (many duplicates/in WIP)
- **151 Markdown files**
- **396 Jupyter notebooks**
- **24 LaTeX files**
- **WIP directory**: 239MB of processed notebooks + many loose files

### Major Issues Identified

1. **Massive WIP Directory** (239MB+)
   - `2025-11-05_ipynb_processed/` - 239MB of converted notebooks
   - Loose files: 100+ PNG images, CSVs, HTML files, notebooks
   - Duplicate template notebooks (Template_ARIMA.ipynb, Template_Darts.ipynb, etc.)
   - Multiple cleanup/conversion scripts

2. **Duplicate Content**
   - Template notebooks in WIP that duplicate Template_*_Python/ structure
   - Code files in both `book/chapters/*/code/` and `WIP/utilities/code/`
   - Multiple versions of same analysis scripts

3. **Unorganized Files**
   - Images, CSVs, HTML files scattered in WIP root
   - Multiple planning documents (CLEANUP_SUMMARY.md, DIRECTORY_STRUCTURE_PLAN.md, etc.)
   - Old conversion scripts that may no longer be needed

4. **Template Structure**
   - 50+ Template directories (good structure, but could be better organized)
   - Some templates have data/ subdirectories, some don't
   - Inconsistent output/ directory usage

## Cleanup Strategy

### Phase 1: Archive & Remove (Low Risk)

1. **Archive Old Processed Notebooks**
   ```bash
   # Move to archive (don't delete yet)
   mkdir -p archive/processed_notebooks
   mv WIP/2025-11-05_ipynb_processed archive/processed_notebooks/
   ```

2. **Consolidate Loose Files in WIP**
   - Move all images to `WIP/assets/images/`
   - Move all CSVs to `WIP/assets/data/`
   - Move HTML files to `WIP/assets/html/`
   - Move Excel files to `WIP/assets/data/`

3. **Remove Duplicate Template Notebooks**
   - Delete `WIP/Template_*.ipynb` files (already have Template_*_Python/ versions)

4. **Consolidate Planning Documents**
   - Merge all planning docs into single `docs/planning/` directory
   - Keep only current/active plans

### Phase 2: Organize WIP (Medium Risk)

1. **Create WIP Structure**
   ```
   WIP/
   ├── active_projects/     # Current work
   ├── archive/             # Old/completed work
   ├── experiments/         # One-off experiments
   ├── assets/              # Images, data, etc.
   └── scripts/             # Utility scripts
   ```

2. **Categorize WIP Subdirectories**
   - Move domain-specific work to `WIP/archive/domain_specific/`
   - Keep only active projects in `WIP/active_projects/`
   - Move completed experiments to `WIP/archive/experiments/`

3. **Clean Up Utilities**
   - Consolidate `WIP/utilities/` and `book/chapters/*/code/`
   - Move shared utilities to `utils/`
   - Archive old/duplicate utilities

### Phase 3: Template Organization (Low Risk)

1. **Standardize Template Structure**
   - Ensure all templates have consistent structure
   - Document template creation process
   - Create template index/registry

2. **Organize by Category**
   ```
   templates/
   ├── forecasting/
   ├── anomaly_detection/
   ├── decomposition/
   ├── econometrics/
   └── ...
   ```

### Phase 4: Documentation (Low Risk)

1. **Create Main README**
   - Clear repository structure
   - How to use templates
   - Contribution guidelines

2. **Template Documentation**
   - Each template should have clear README
   - Usage examples
   - Dependencies

## Immediate Actions (Safe to Do Now)

### 1. Create Archive Structure
```bash
mkdir -p archive/{processed_notebooks,old_scripts,old_plans}
mkdir -p WIP/assets/{images,data,html}
mkdir -p docs/planning
```

### 2. Move Loose Files
```bash
# Images
find WIP -maxdepth 1 -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" | xargs -I {} mv {} WIP/assets/images/

# Data files
find WIP -maxdepth 1 -name "*.csv" -o -name "*.xlsx" | xargs -I {} mv {} WIP/assets/data/

# HTML files
find WIP -maxdepth 1 -name "*.html" | xargs -I {} mv {} WIP/assets/html/
```

### 3. Remove Duplicate Templates
```bash
# Remove template notebooks from WIP (we have Python versions)
rm WIP/Template_*.ipynb
```

### 4. Consolidate Planning Docs
```bash
mv WIP/*.md docs/planning/ 2>/dev/null || true
# Keep only active ones in root
```

## Long-term Organization

### Proposed Structure
```
time_series/
├── templates/              # All Template_*_Python/ moved here
│   ├── forecasting/
│   ├── anomaly_detection/
│   └── ...
├── experiments/           # Active experiments
├── data/                  # Shared data
├── utils/                 # Shared utilities
├── docs/                  # Documentation
│   ├── planning/
│   └── guides/
├── book/                  # Book content
├── WIP/                   # Work in progress (organized)
│   ├── active_projects/
│   ├── archive/
│   └── assets/
└── archive/               # Historical/old content
```

## Risk Assessment

- **Low Risk**: Moving files to archive, organizing WIP
- **Medium Risk**: Consolidating utilities, reorganizing templates
- **High Risk**: Deleting files (should be done carefully with backups)

## Next Steps

1. Review this plan
2. Start with Phase 1 (safest)
3. Test changes in a branch
4. Gradually proceed through phases

