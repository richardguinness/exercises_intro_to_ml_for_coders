# A Random Forest from Scratch

As per fast.ai ML for Coders 2018 course, lesons 6 & 7.

The body of Jeremy's solution can be found in lesson3-rf_foundations.ipynb.

&copy; Richard Guinness August 2022

(Please see my separate article which relates to this article: [Understanding the place of `min_leaf` in Jeremy Howard's algorithm](https://richardguinness.github.io/2022/08/02/min_leaf.html))

Having established my first effort (a class named `TreeEnesembleM`):

```python
>>> ensM = TreeEnsembleM(X_train, y_train, 10, sample_sz=200, min_leaf=5)
>>> predsM = ensM.predict(X_valid.values)
>>> predsM[:10] # NOTE: We've taken a slice of values

array([0.25619048, 0.05095238, 0., 0.95606061, 0.8000974 , 0., 0.07838384, 0.83492063, 0., 0.36852814])
```

Jeremy's solution (when using the exact same parameters) produces the following results.

```python
array([0.29571429, 0.09095238, 0., 0.9, 0.76190476, 0., 0.05333333, 0.88380952, 0., 0.35587302])
```

What's causing this difference?

Also, I noticed that Jeremy's code doesn't pass min_leaf on to new nodes, but this issue is circumvented because I have specified `min_leaf` 5 (during these tests) which is the default of a new DecisionTree object.

By transplanting my function definitions into Jeremy's body of code, I established the following:

__Importance of slicing x & y with an appropriate index__

Instead of copying x & y (which are quite large variables) during the creation of each node, we should instead simply collate a list of indices relevant to each node. In my original code, I pass the whole lot to each node on instantiation. A left/right mask is created all over again on the whole lot, and this in some instances results in a recursion depth error.

To avoid this eventuality, I needed to update my declaration of x & y within `find_better_split()` as follows:

From:

    x = self.x.iloc[:,var_idx].values
    y = self.y
    
to:

    x = self.x.iloc[self.idxs,var_idx].values
    y = self.y[self.idxs]

I also established that there were some issues in the rest of my `find_better_split()` algorithm.

    if m_left.sum() <= self.min_leaf or m_right.sum() <= self.min_leaf: # We've found a leaf
        continue

The following modifications were necessary:


    if m_left.sum() < self.min_leaf or m_right.sum() < self.min_leaf: # We've found a leaf
        continue

However, I'm not clear that this change is correct. Surely we want to interrupt this loop if we find a group equal in size to `min_leaf`? This could be investigated by comparing each models results with say SKlearn's implementation...

## Findings/Takeaways

1. Realised that transplanting functions between Jeremy & my versions was possible. Each function should achieve the same effect.
2. Realised that comparing the performance of one function with another would be informative (see "Testing `find_better_split()` below". This proved quite cumbersome to do: I literally copy and pasted the function body to a new cell, synthesised a data set, established new variables, and eventually this lead to the discovery that the `m_left.sum() <= self.min_leaf` was responsible for a fishy outcome. However, while I established a solution in this exercise, when implemented in the main program the goal wasn't achieved... this leads me to believe that Jeremy's handling of `min_leaf` probably needs a little exploration.