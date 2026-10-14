# Candidate Generators
In the second week you will work with candidate generation in `engine`, as well as with model training and validation algorithms in `engine_builder`.
- Start with the [task about quality metrics](./metrics/README.md).
- [Learn](./user_item_matrix/README.md) how to build a user-item matrix for factorization.
- Implement two matrix factorization algorithms - [BPR](./bpr/README.md) and [Explicit ALS](./explicit_als/README.md).
- Implement an [EmbeddingStorage](./embedding_storage/README.md) for the runtime on top of the Qdrant vector DB and the [SLIM](./slim/README.md) candidate generator.
- Implement the offline process for training [Implicit ALS](./implicit_als/README.md) and a CG based on EmbeddingStorage.
- Implement a [component for the iALS step](./runtime_als/README.md) towards the user in the runtime.

When you are done, you will have an impressive set of different candidate generators and different algorithms for training them to experiment with for your recommender - for example, target reweighting applies to different models, and an example `evaluation` script for ImplicitALS is in the root of `engine_builder`.
