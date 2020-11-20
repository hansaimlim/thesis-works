## Data set description
### `LCO` means Leave Chemical Out
1. pKd and pKi activities are combined `combined activity`
2. Invalid (e.g. NaN, pKd=0) or inequality activities are removed  from `combined activity`
3. Activities for non-kinase proteins are removed from `combined activity`
4. From the activity data above, a small number of unique chemicals were chosen for `dev/test`
5. Any other chemicals similar (Tanimoto similarity >=0.8) to the chosen `dev/test` chemicals are also collected for `dev/test`
6. Thus, for all `training` chemicals, __maximum__ Tanimoto similarity to any `dev/test` chemical is __0.8__
7. Activities regarding `dev/test` chemicals (against any protein) are held out for `dev` and `test` data
8. The rest of activities are for `train` data


## File description
Filename | Description
----------|----------
trainpairs | `train data` described above (__LCO__)
devpairs | `dev data` described above (__LCO__)
testpairs | `test data` described above (__LCO__)
