# experiments_server

A service for managing experiments and splitting users into groups by hashing.

## Endpoints

- `PUT /experiments/{experiment_name}`: Create or update an experiment.
- `GET /experiments/{experiment_name}/resolve?user_id=123`: Get the parameter schema of a specific experiment for a user.

## Storage model
- All experiments are stored in SQLite.
- Each experiment has a salt, rules for splitting into groups and default parameters.
- Each group has overrides of the default parameters (if any) and a name.
- When parameters are requested for a user, the ID is hashed, and the group and the parameters with the override applied are returned.
