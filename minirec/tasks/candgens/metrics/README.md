# Offline Metrics

## About the task
Offline, when training models, we want to measure their quality, and we want to do it consistently: one method and the same data for different models or even whole pipelines. We also need some integration approach that imitates the real recommender as closely as possible, including data reloading and passing the context.

Writing such logic is a laborious task! We will start simple: implementing metrics.

## Statement
Implement the quality metrics of recommendation algorithms in [`engine_builder/tasks/evaluation/metric.py`](../../../engine_builder/tasks/evaluation/metric.py), following the formulas below. For some metrics you will need to precompute statistics or get additional information: at initialization you will have access to past actions or to information about each item as a polars dataframe.

Notation:  
- $\hat{R}_u$ is the list of candidates for user $u$ (in ranking order)
- $\hat{R}_{u,k}$ is the first $k$ candidates  
- $R_u$ is the set of positive objects  
- $|\cdot|$ is the cardinality of a set  
- $\mathbb{1}[\cdot]$ is the indicator  
- $\log$ is the natural logarithm

### Ranking metrics
Implement the following metrics:

$$
\mathrm{APK@k}(u)=\frac{1}{\min(|R_u|,k)}\sum_{i=1}^{k}\frac{\sum_{j=1}^{i}\mathbb{1}[r_j\in R_u]}{i}\cdot \mathbb{1}[r_i\in R_u]
$$

$$
\mathrm{MAP@k}=\frac{1}{|U|}\sum_{u\in U}\mathrm{APK@k}(u)
$$

$$
\mathrm{DCG@k}(u)=\sum_{i=1}^{k}\frac{\mathbb{1}[r_i\in R_u]}{\log(i+1)}
$$

$$
\mathrm{NDCG@k}(u)=\frac{\mathrm{DCG@k}(u)}{\mathrm{IDCG@k}(u)}
$$
where $\mathrm{IDCG@k}(u)$ is the DCG of the ideal ranking, in which all positives are at the top:
$$
\mathrm{IDCG@k}(u)=\sum_{i=1}^{\min(|R_u|,k)}\frac{1}{\log(i+1)}
$$
The ideal is built over all positives $R_u$, not only those that made it into the output: a list that found one positive out of three does not get $\mathrm{NDCG}=1$. Without `k`, take $k = |\hat{R}_u|$.

### Serendipity@k
Let $H_u$ be the user's history (objects already seen); for a user without actions $H_u=\varnothing$. Implement:
$$
\mathrm{Serendipity@k}(u)=
\frac{\sum_{r\in \hat{R}_{u,k}} \mathbb{1}[r\in R_u \land r\notin H_u]}
{\sum_{r\in \hat{R}_{u,k}} \mathbb{1}[r\in R_u]}
$$

### CategoryDiversity@k
Let $\mathrm{cat}(i)$ be the category of object $i$. Implement:
$$
\mathrm{CategoryDiversity@k}(u)=
\frac{|\{\mathrm{cat}(r)\;|\;r\in \hat{R}_{u,k}\}|}{|\hat{R}_{u,k}|}
$$

## How to check yourself
```bash
(cd engine_builder && uv run pytest tests/evaluation/test_metric.py)
```
