# YSDA Recommender Systems Course

This repository contains materials for the Recommender Systems course taught at the [Yandex School of Data Analysis](https://shad.yandex.ru). This branch corresponds to the ongoing 2026 spring semester.

## Syllabus

- Week 1: Intro
    - Lecture: course overview and organizational details, intro to "Recommender Systems" problem
    - Practice: basic recommenders, user-item latent space
- Week 2: Candidate generation & metrics
    - Lecture: recsys metrics & candidate generation: classic ML, ANN, mixing
    - Practice: yambda contest overview & baseline solution
- Week 3: Ranking, diversity & metrics
    - Lecture: reranking - losses, algorithms, metrics; diversity control and MRR / DPP
    - Practice: classic algorithms (MF, SLIM, EASE); ranking - pool building, undersampling, composite targets
- Week 4: Deep learning for RecSys & neural candidate generation
    - Lecture: two-tower architecture, softmax model & sampled softmax loss, contrastive learning, negative sampling and LogQ
    - Practice: paper review on LogQ correction and negative sampling techniques
- Week 5: Neural candidate generation, pt.2
    - Lecture: cold start and long-tail, user / item encoding strategies, sequential models, beyond two-tower (GPU retrieval, generative retrieval)
    - Practice: aspects of training neural networks for RecSys
- Week 6: Neural ranking, pt.1
    - Lecture: why use DL, feature encoding (categorical, embedding, scalar), model composition
    - Practice: paper review on Piecewise Linear Encoding and Unified Embeddings
- Week 7: Neural ranking, pt.2
    - Lecture: feature interaction modelling, MLP, multi-task & knowledge distillation
    - Practice: paper review on DCNv2 architecture 
- Week 8: System Design, pt. 1
    - Lecture: data architectures, logging, biases, data drift and monitoring
    - Practice: feature storages for different scales
- Week 9: System Design, pt. 2
    - Runtime design, candidate funnel, GPU inference, data delivery, controlled degradation, cold start
    - Practice: paper review on GPU retrieval - LiNR, SilverTorch
- Week 10: RecSys Transformers applications
    - Lecture: all about sequential models (transformers) on user action history
    - Practice: paper review on PinnerFormer & TransAct (by Pinterest)
- Week 11: Reinforcement Learning in RecSys
    - Lecture: all about bandits - algorithms, Thompson sampling, contextual bandits
    - Practice: applications of bandits & off-policy evaluation for e-grocery
- Week 12: Case Studies of Yandex's services
    - Lecture: recsys design for e-grocery, movie streaming platform, blogpost & news feed
    - Practice: deep dive in Lavka (Yandex's e-grocery) recommender system
- Week 13: Trends in RecSys
    - Lecture: state-of-the-art approaches & novel papers in RecSys
    - Practice: best contest colutions on yambda retrieval

## Homeworks

- Week 1: [minirec: basics and services](minirec/tasks/basics_and_services/) - storages, simple components & service runtime
- Week 2: [minirec: candidate generators](minirec/tasks/candgens/) - metrics, classic models (ALS, BPR, SLIM), vector indexes
- Week 3: [minirec: ranking](minirec/tasks/ranking/) - candidate logging, training pool construction, model ranker, Mixigen
- Week 4: [two-tower models](week04_dl_candgen/homework/homework.ipynb) - metrics, loss functions and negative sampling for candidate generation
- Week 7: [neural ranking](week07_dl_ranking/homework/homework.ipynb) - feature encoding and a multi-task DCN V2 ranker with a mixture of experts
- Week 10: [transformers for candidate generation](week10_transformers/homework/homework.ipynb) - dataset, architecture, training and inference
- Week 13: [Semantic IDs](week13_trends/homework/homework.ipynb) - bonus homework on generative retrieval

## Staff

- [Daniil Tkachenko](https://github.com/deadpadre)
- [Kirill Khrylchenko](https://github.com/KhrylchenkoKirill)
- [Roma Nigmatullin](https://github.com/rmnigm)
- [Alexey Krasilnikov](https://github.com/KrasilnikovAV)
- [Artem Matveev](https://github.com/matfu-pixel)
- [Vladimir Baikalov](https://github.com/NonameUntitled)
- [Vlad Tytskiy](https://github.com/Tytskiy)
- [Ruslan Kuliev](https://github.com/kuliev-r)
