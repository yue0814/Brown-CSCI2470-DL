Assignment 6: Sequence to Sequence Machine Translation
====
Yue Peng
----

In my first step, I initialized some hyperparameters, like learning rate = 1e-3 (the same as last assignment), and window size = 12, RNN size = 64 (too large will caused lower speed), embedding size = 30. During debugging, I successively found out the suqence_mask problem (never used it) and dropout problem(set as 0.5). 