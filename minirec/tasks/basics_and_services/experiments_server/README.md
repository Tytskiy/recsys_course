# Experiments Server

## About the task
In recommender systems we often want to run experiments, A/B tests in particular: split users into two (or more) groups in some random way and apply different system parameters / different algorithms depending on the group they fall into. One common way to do this is to hash user identifiers and split them into buckets, mapping each bucket to some group in the experiment. This service provides exactly that functionality: create an experiment and describe its groups, describe the default parameters and the overrides of these parameters in each group.

## Statement
Implement an API for storing and managing experiments. The provided template uses FastAPI; the pydantic schemas for the methods are already given to you in `models.py`, as is the hash function `salted_hash`, which determines the bucket number from 1 to `mod` from the identifier and the experiment salt. Use them.

You need to implement two methods and the ExperimentRepository class. When the description and the tests contradict each other, always follow the tests, which you can find in the `tests` folder and run locally from the directory:
```bash
(cd experiments_server && uv run pytest)
```

## Methods
### `PUT /experiments/{experiment_name}`
The method for creating an experiment; it accepts JSON with the `ExperimentDefinition` schema. It creates and stores a new experiment in the DB and validates it: the groups must cover all buckets from 1 to `mod`, the groups must not overlap, and there must be no buckets that fall into no group. The order of groups in the request is arbitrary. A repeated `PUT` with the same name fully replaces the experiment (salt, `mod`, `default`, groups).

Partition errors return 422 with `{"detail": <message>}`; the checks run in this order:
1. empty list of groups: `"groups must be non-empty"`;
2. the first group does not start at 1 or the last one does not end at `mod`: `"groups must fully cover the modulo range"`;
3. a gap or an overlap between adjacent groups: `"groups must be contiguous"`.

### `GET /experiments/{experiment_name}/resolve`
The method for getting the user's group in the experiment and the parameters for that group; it accepts the integer parameter `user_id`, the user identifier. The bucket is computed through `ExperimentRepository.salted_hash` with this experiment's salt and `mod` (the tests replace this method, so call exactly it); the group is the one with `range_start <= bucket <= range_stop` (both bounds inclusive). It returns the group name and `default | overrides`. For an unknown experiment: 404 with `{"detail": "Experiment '<name>' not found"}`.


## Tips
- You can use SQLModel and SQLAlchemy as the ORM; they integrate well with FastAPI.
- Use the JSON type to store parameters and overrides: we do not know in advance what structure the parameters will have.
- To combine `default` and `override`, if a parameter is a dict or a list, it is enough to overwrite parameters entirely at the first level.
- Updating an experiment is simply creating a new experiment with the same name.
- The schema from `models.py` already rejects `mod <= 0` and groups with `range_stop < range_start` (422); range coverage, contiguity and non-overlapping of groups are up to you.

## About the salt
User identifiers for A/B testing are hashed with some "salt", a set of characters / bytes unique to the experiment. The salt is needed for several reasons:
- Randomizing the split by setting nothing but the salt.
- The ability to run many experiments at the same time (although at scale this should be checked additionally).
- Changing the salt between consecutive experiments removes the carryover effect.

Industrial A/B testing platforms have many more tools for advanced splitting: for example, the [Avito](https://habr.com/ru/companies/avito/articles/454164/) platform also creates layers to control which experiments may and may not overlap.
