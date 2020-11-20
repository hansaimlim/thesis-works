## Directory for matlab-readable matrices

## File description
Filename | Description
----------|----------
binarize_activity.py | python script to binarize integrated complete data set
integrated_active.tsv | list of active pairs
integrated_inactive.tsv | list of inactive pairs
multirecord_pairs.dat.gz | list of pairs with _multiple records._ Binarizer scores and decisions are shown.

* For binarization scheme, please read the _comments_ in __binarize_activity.py__ script
* Basic idea is to give more points for strong potency, deduct more for very low activity. If overall positive, the pair is considered active.
* Decisions for multi-record pairs are available in __multirecord_pairs.dat.gz__

