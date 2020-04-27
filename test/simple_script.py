#!/usr/bin/env python
"""Exercise some simple functions to make sure their citations are picked up"""
import numpy as np
import bct

# Zachary karate club http://konect.cc/networks/ucidata-zachary/
s = """
1 2
1 3
2 3
1 4
2 4
3 4
1 5
1 6
1 7
5 7
6 7
1 8
2 8
3 8
4 8
1 9
3 9
3 10
1 11
5 11
6 11
1 12
1 13
4 13
1 14
2 14
3 14
4 14
6 17
7 17
1 18
2 18
1 20
2 20
1 22
2 22
24 26
25 26
3 28
24 28
25 28
3 29
24 30
27 30
2 31
9 31
1 32
25 32
26 32
29 32
3 33
9 33
15 33
16 33
19 33
21 33
23 33
24 33
30 33
31 33
32 33
9 34
10 34
14 34
15 34
16 34
19 34
20 34
21 34
23 34
24 34
27 34
28 34
29 34
30 34
31 34
32 34
33 34
""".strip()

arr = np.zeros((34, 34), dtype=np.uint8)
for row in s.split('\n'):
    first, second = row.split(' ')
    arr[int(first)-1, int(second)-1] += 1

arr = bct.binarize(arr + arr.T)

np.random.seed(1991)

eff = bct.efficiency_bin(arr)
mod = bct.modularity_und(arr)
rand = bct.randmio_und_connected(arr, 5)