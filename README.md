# Disclex 

Initial alignment format (CSV with header). Note that because this is a CSV, if
a phone/phoneme being used includes a comma, each field must be quoted. By
invisible, we mean invisible to the Dirichlet process for purpose of grouping,
not invisible as part of the FST grammar.

```
parameter,observation,split
f u,f ə,ug
b a r,b a r,ident
b a z,b a z,train
w ə n,w ə n,test
```

Split file:

```
split,status,mix
train,fixed,1
test,trainable,1
ug,invisible,0.45
ident,invisible,80
```

[//]: <> TODO(padril): Describe the difference between the statuses.

