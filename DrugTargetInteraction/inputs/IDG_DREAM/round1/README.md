## Data set description
1. pKd and pKi activities are combined `combined activity`
2. Invalid (e.g. NaN, pKd=0) or inequality activities are removed  from `combined activity`
3. Activities for non-kinase proteins are removed from `combined activity`
4. Chemicals whose Tanimoto similarity > 0.8 to any of Round 1 chemicals are held out as `round1 chemicals`
5. All activities regarding `round1 chemicals` are separated for dev/test data, namely `holdout data`
6. Randomly split approximately 80%/20% from `holdout data` to prepare `dev/test data`
7. The rest of `combined activity` is `train data`
8. Round 1 data obtained from IDG DREAM Challenge `https://www.synapse.org/#!Synapse:syn15667962`
9. Since too few samples with activity>=7.0 , such pairs for other chemicals are _additionally_ held out for dev/test `v2`
10. Data `_v2` contains 566125 `train_v2`, 21920 `dev_v2`, and 3499 `test_v2` samples

## File description
Filename | Description
----------|----------
trainpairs | `train data` described above (plain version, deprecated)
devpairs | `dev data` described above (plain version, deprecated)
testpairs | `test data` described above (plain version, deprecated)
trainpairs_v2 | `train_v2` described above
devpairs_v2 | `dev_v2` described above
testpairs_v2 | `test_v2` described above
round1pairs | Round 1 data. Activity unknown
