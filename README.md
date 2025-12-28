# gaming-lifetimevalue

## Quick Start

```bash
git clone <repo-url>
cd gaming-lifetimevalue
make install
```

For development:
```bash
make install-dev
```

## Requirements

MacOS with M-series CPU needs libomp for lightgbm:
```bash
brew install libomp
```


## What I would do next/with more time

### For production robustness
- Implement data preprocessing in dbt for better readability and automatic lineage and quality tests
- Add strong type checking and column schema to input and output dataframes (pydantic, pandera)

### For ML performance
- Increase training dataset size by infering `120d_rev' from recent horizons (up to horizon 30 or 60)
