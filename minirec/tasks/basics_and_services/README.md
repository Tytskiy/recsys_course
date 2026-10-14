# Basics and services
In the first week you will get to know the code, study its structure and interfaces, and implement yourself:
- The key-value feature storage component `ValkeyFeatureStorage`.
- Two simple candidate generation algorithms - `TopPop` and `ItemToItemLists`.
- An experiment management service.
- The API of the recommendation service (and a bonus task - a runtime with file reloading).

The tasks can be solved in any order: the tests of each task replace the code of other tasks with fakes (the storage in the item lists, the runtime in the service API). To run the whole service you need all the tasks except the bonus reloading; the Valkey-based storage is a convenient place to start.

As a result of these tasks you will have the first working version of the service, which you can run yourself and experiment with through the web interface; the instructions for this are in the corresponding modules, although this is not required for the tasks.
