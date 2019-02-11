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

`march=native` optimize selon la machine qui fait tourner le code

`fno-tree-vectorize` pas de vectorisation automatique

`fopt-info` verbose sur la vectorisation


### Moyenne 3 vecteurs: 
scalaire : on prends la moyenne terme à terme
vectorielle : on prends la moyenne sur un vecteur de 8 termes (__mm256) (on somme et on divise par 3)

Resultat: 
- sans vectorisation automatique (optimization 2)
```
g++ -o exe/test.exe obj/main.o -O2 -std=c++11 -march=native -fno-tree-vectorize  -Iinclude   -lm
size : 1048576
temps scalaire 7.92527e-10
temps vectoriel 7.94073e-10
```
- sans vectorisation automatique (optimization 0)
```
g++ -o exe/test.exe obj/main.o -O0 -std=c++11 -march=native -fno-tree-vectorize  -Iinclude   -lm
size : 1048576
temps scalaire 2.46212e-09
temps vectoriel 2.52547e-09
```

- avec vectorisation automatique 
```
g++ -o exe/test.exe obj/main.o -O3 -std=c++11 -march=native -Iinclude   -lm
size : 1048576
temps scalaire 4.81963e-10
temps vectoriel 4.85291e-10
```

- avec vectorisation verbose
```
g++ -o exe/test.exe obj/main.o -O3 -std=c++11 -march=native -ftree-vectorize -fopt-info -Iinclude   -lm
size : 1048576
temps scalaire 4.70113e-10
temps vectoriel 4.75479e-10
```
On constate qu'il n'y a très peu de difference entre le temps pris par un calcul scalaire ou un calcul vectoriel. 

### Produit scalaire

```
g++ -o exe/test.exe obj/main.o -O2 -std=c++11 -march=native -fno-tree-vectorize  -Iinclude   -lm
size : 1048576
temps scalaire 9.67618e-10
temps vectoriel 9.76482e-10
```


### valeur absolue

```
g++ -o exe/test.exe obj/main.o -O2 -std=c++11 -march=native -fno-tree-vectorize  -Iinclude   -lm
size : 1048576
temps scalaire 3.52069e-10
temps vectoriel 1.86498e-10
``` 

absorption : 

10^7 + 10^-7 - 10^7
 =/=
10^7 - 10^7 + 10^-7

loop unrolling https://en.wikipedia.org/wiki/Loop_unrolling
