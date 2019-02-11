Makefile flags

```
# -- Flags ----------
# PAS DE VECTORISATION -----
# C_OPTIMISATION_FLAGS = -O2 -std=c++11 -march=native -fno-tree-vectorize 

# VECTORISATION INVISIBLE
# C_OPTIMISATION_FLAGS = -O3 -std=c++11 -march=native

# VECTORISATION FORCEE
# C_OPTIMISATION_FLAGS = -O3 -std=c++11 -march=native -ftree-vectorize -fopt-info
```

`-march=native` optimize selon la machine qui fait tourner le code

̀`-fno-tree-vectorize` pas de vectorisation automatique


̀`-fopt-info` verbose sur la vectorisation

