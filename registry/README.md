# Community Fairness Configuration Registry

This directory contains published community fairness configurations — machine-readable files that encode what a specific community defined as "fair" for a specific decision domain.

## How It Works

1. A community runs a [facilitation session](../docs/community_input_protocol.md) and produces a configuration
2. The configuration is published here, organized by jurisdiction and domain
3. Any organization can check their data against a published config using the `/audit/compliance` API endpoint
4. Each config has a UUID, provenance record, and expiration date

## Directory Structure

```
registry/
├── README.md
├── index.json                          # Machine-readable index of all configs
├── michigan/
│   ├── lending/
│   │   └── detroit_2026.json           # Detroit community config for lending
│   └── employment/
│       └── detroit_2026.json           # Detroit community config for hiring
├── georgia/
│   └── lending/
│       └── atlanta_2026.json
└── templates/
    └── blank_config.json               # Starting template for new configs
```

## Using a Published Config

### Via API
```bash
CONFIG=$(cat registry/michigan/lending/detroit_2026.json)
curl -X POST https://your-api/audit/compliance \
  -F "file=@your_data.csv" \
  -F "race_col=race" \
  -F "outcome_col=approved" \
  -F "favorable_value=1" \
  -F "config_json=$CONFIG"
```

### Via Python
```python
from integrations.aif360_adapter import CommunityAIF360Audit

audit = CommunityAIF360Audit.from_config("registry/michigan/lending/detroit_2026.json")
results = audit.run(df, label_col="action_taken", protected_col="derived_race")
```

## Publishing a New Config

1. Run a community session following the [protocol](../docs/community_input_protocol.md)
2. Generate the config using `community_input.build_community_config()`
3. Validate against the [CDF v1.0 schema](../specs/community_fairness_config_v1.schema.json)
4. Place in the appropriate jurisdiction/domain directory
5. Update `index.json`
6. Submit a pull request

## Validation

All configs in this registry must conform to the CDF v1.0 schema and pass validation:

```bash
python -m registry.validate
```

## License

Community configurations are published under CC-BY-SA 4.0. Communities retain authorship credit for their configurations.
