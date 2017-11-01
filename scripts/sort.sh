#!/bin/bash
# sort RDF data in predicate order

for f in id_uni*.nt; do
    fa=$f
    fb=${f/nt/nt_sorted}
    echo "sort -n -k 2 $fa > $fb"
    sort -n -k 2 $fa > $fb
done

