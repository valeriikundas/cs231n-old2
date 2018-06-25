# Lecture 3 Loss Functions and Optimization

## Multiclass SVM loss:
Loss[i] = sum(j!=y[i])( max(0, s[j] - s[y[i]] + 1) )
