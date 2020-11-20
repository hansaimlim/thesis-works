## Data set description
1. pKd and pKi activities are combined `combined activity`
2. Invalid (e.g. NaN, pKd=0) or inequality activities are removed  from `combined activity`
3. Activities for non-kinase proteins are removed from `combined activity`
4. Approximately 4.5% data for dev/test pairs, making sure no overlap across data set
5. No special data clustering applied (e.g. by chemical similarity, etc.)

## File description
Filename | Description
----------|----------
trainpairs | `train data` described above
devpairs | `dev data` described above
testpairs | `test data` described above
