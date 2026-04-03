## Summary

Describe the change and the problem it solves.

## Scope

- Data pipeline
- Model architecture
- Losses
- Training or distributed training
- CLI or notebook workflow
- Documentation

## Validation

List the checks you actually ran.

```bash
# Example:
python3 -m pytest --capture=no tests -q
```

## Notes for Reviewers

Call out anything that deserves extra attention:

- risky tensor-shape changes
- numerics or masking behavior
- dataset assumptions
- GPU or distributed assumptions
- follow-up work not included here

## Checklist

- [ ] The change is focused and intentionally scoped
- [ ] Tests were added or updated when behavior changed
- [ ] Relevant docs, configs, or notebooks were updated
- [ ] Large generated artifacts, datasets, or checkpoints were not added
- [ ] Security-sensitive changes were reviewed carefully
- [ ] I ran the validation commands listed above
