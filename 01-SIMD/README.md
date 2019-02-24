## Programmation Parallèle TP 1 - SIMD

### Goal :

The main goal of this TP is to take a look at how we can use `simd` to vectorize programs. The code is written in `C++` and complied using the following flags.

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

`march=native` is an option that tells the compiler what code should be produced for the system's processor architecture.

`-ftree-vectorize` is an optimization option which attempts to vectorize loops using the selected ISA if possible.

`fno-tree-vectorize` is an option to not use automatic vectorisation that happens when setting `-03`.

`fopt-info` more verbose for the vectorisation.

####Link to documentation :
`https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_loadu_ps&expand=3377`

####Useful Functions :
`_mm256_loadu_ps(&A[i])` : loads the values starting from A[i] into the m256
`_mm256_add_ps(a, b)` : adds two `__m256` vectors
`_mm256_storeu_ps` : stores the `__m256` vector into a normal array.
`_mm256_dp_ps` : computes the dot product for two `__m256` vectors. The third term is a mask to specify how the answer is given. (0000 0000) The first 4 bits indicates which elements are taken into account for the dot product (1111, means you multiply and add everything). The last 4 bits indicate where the results are stored. It returns a `__m256` vector.(X,X,X,X,Y,Y,Y,Y). X is the dot product for the first 4 values of the `__m256` and Y for the last 4. To get the total dot product, you need to add `X` and `Y`.

#### Notes:

- Remember to free the allocated space for the vectors.
- Absorption : $10^7 + 10^-7 - 10^7 \neq 10^7 - 10^7 + 10^-7$

### Average of 3 vectors:

**Scalar** : on prends la moyenne terme à terme

```C
for (unsigned long j = 0; j < size; j++)
{
    M[j] = (A[j] + B[j] + C[j]) / 3;
}
```

**Vectorial** : on prends la moyenne sur un vecteur de 8 termes (\_\_mm256) (on somme et on divise par 3)

```C
for (int i = 0; i < size; i += 8)
{
    __m256 a = _mm256_loadu_ps(&A[i]); // loads the values starting from A[i] into the m256
    __m256 b = _mm256_loadu_ps(&B[i]);
    __m256 c = _mm256_loadu_ps(&C[i]);
    // add a and b to m
    __m256 m = _mm256_add_ps(a, b);
    // add c and m to m
    m = _mm256_add_ps(m, c);
    // set a variable to 3 (we would like to divide by 3 for the average)
    a = _mm256_set1_ps(3);
    // division
    m = _mm256_div_ps(m, a);

    // mettre le resultat dans le vecteur global
    _mm256_storeu_ps((float *)(M_simd + (i * 8)), m);
}
```

Results:

- without automatic vectorisation (optimization 3)

```
g++ -O3 -std=c++11 -march=native -fno-tree-vectorize  -Iinclude -c src/main.cpp -o obj/main.o
g++ -o exe/test.exe obj/main.o -O3 -std=c++11 -march=native -fno-tree-vectorize  -Iinclude   -lm
size : 1048576
temps scalaire 1.50843e-09
temps vectoriel 1.7334e-09
The result is coherent
```

- with automatic vectorisation

```
g++ -O3 -std=c++11 -march=native -Iinclude -c src/main.cpp -o obj/main.o
g++ -o exe/test.exe obj/main.o -O3 -std=c++11 -march=native -Iinclude   -lm
size : 1048576
temps scalaire 1.35489e-09
temps vectoriel 1.38454e-09
The result is coherent
```

- with vectorisation verbose

```
g++ -O3 -std=c++11 -march=native -ftree-vectorize -fopt-info -Iinclude -c src/main.cpp -o obj/main.o
src/main.cpp:55:37: note: loop vectorized
src/main.cpp:55:37: note: loop with 6 iterations completely unrolled (header execution count 46365075)
g++ -o exe/test.exe obj/main.o -O3 -std=c++11 -march=native -ftree-vectorize -fopt-info -Iinclude -lm
size : 1048576
temps scalaire 2.0545e-09
temps vectoriel 2.5712e-09
The result is coherent
```

There's little to no difference between the time taken to compute for the scalar or vectorial version. The vectorial version is consistently slightly slower than the scalar one.

#### Loop Unrolling

Noticed during the verbose vectorisation
https://en.wikipedia.org/wiki/Loop_unrolling

e.g :

```c
int x;
 for (x = 0; x < 100; x++)
 {
     func(x);
 }
```

If the compiler knows that the function can be vectorized.
`x -> [x, x+1, x+2, x+3, x+4]`

```c
 int x;
 for (x = 0; x < 100; x += 5 )
 {
     func(x);
     func(x + 1);
     func(x + 2);
     func(x + 3);
     func(x + 4);
 }
```

By unrolling the loop, the program requires more space in memory to execute the loop. However, the speed of calculation is improved.

### Scalar product

We would like to calculate
$A.B = \sum a_ib_i $

**Scalar** :

```c
    S = 0;

    for (int j = 0; j < size; j++)
    {
        S += A[j] * B[j];
    }
```

**Vectorial** : There is a dot product function `_mm256_dp_ps` in SIMD that computes the scalar product for 2 vectors.

- sans vectorisation automatique (optimization 3)

```
g++ -O2 -std=c++11 -march=native -fno-tree-vectorize  -Iinclude -c src/main.cpp -o obj/main.o
g++ -o exe/test.exe obj/main.o -O2 -std=c++11 -march=native -fno-tree-vectorize  -Iinclude   -lm
size : 1048576
temps scalaire 4.57764e-14
temps vectoriel 2.40665e-09
The result is coherent
```

- avec vectorisation verbose

```
g++ -O3 -std=c++11 -march=native -ftree-vectorize -fopt-info -Iinclude -c src/main.cpp -o obj/main.o
/usr/local/include/c++/8.2.0/bits/locale_facets.h:877:23: note: speculatively devirtualizing call in std::ctype<char>::char_type std::ctype<char>::widen(char) const/7123 to virtual std::ctype<char>::char_type std::ctype<char>::do_widen(char) const/1410
src/main.cpp:54:37: note: loop vectorized
src/main.cpp:54:37: note: loop with 6 iterations completely unrolled (header execution count 46365075)
g++ -o exe/test.exe obj/main.o -O3 -std=c++11 -march=native -ftree-vectorize -fopt-info -Iinclude   -lm
size : 1048576
temps scalaire 3.8147e-14
temps vectoriel 1.24539e-09
The result is coherent
```

### MinMax

**Scalar** :

```C
min = FLT_MAX;
max = FLT_MIN;
t0 = std::chrono::high_resolution_clock::now();

for (unsigned long j = 0; j < size; j++)
{
    if (min > A[j])
    {
        min = A[j];
    }
    if (max < A[j])
    {
        max = A[j];
    }
}
```

**Vectorial** :

```C
max_simd = FLT_MIN;
__m256 local_min = _mm256_set1_ps(FLT_MAX);
__m256 local_max = _mm256_set1_ps(FLT_MIN);

// each simd vector is size 8, we need to split the original vecteur into the appropriate size
for (int i = 0; i < size / 8; i++)
{
    __m256 a = _mm256_loadu_ps(&A[i * 8]);
    // minimum term by term of the two vectors.
    local_min = _mm256_min_ps(a, local_min);
    local_max = _mm256_max_ps(a, local_max);
}

// store the mm256 into an array and iterate over it to find the max and the min

float *min_vec = (float *)malloc(8 * sizeof(float));
float *max_vec = (float *)malloc(8 * sizeof(float));
_mm256_storeu_ps((float *)(min_vec), local_min);
_mm256_storeu_ps((float *)(max_vec), local_max);

// iteration
for(int i=0; i<8; i++)
{
    if (min_vec[i] < min_simd)
    {
        min_simd = min_vec[i];
    }

    if (max_vec[i] > max_simd)
    {
        max_simd = max_vec[i];
    }
}
```

- sans vectorisation automatique (optimization 3)

```
g++ -O2 -std=c++11 -march=native -fno-tree-vectorize  -Iinclude -c src/main.cpp -o obj/main.o
g++ -o exe/test.exe obj/main.o -O2 -std=c++11 -march=native -fno-tree-vectorize  -Iinclude   -lm
Temps scalaire : 1.11263e-09
Temps vectorielle : 2.1708e-10
```

- avec vectorisation verbose

```
g++ -O3 -std=c++11 -march=native -ftree-vectorize -fopt-info -Iinclude -c src/main.cpp -o obj/main.o
/usr/local/include/c++/8.2.0/bits/locale_facets.h:877:23: note: speculatively devirtualizing call in std::ctype<char>::char_type std::ctype<char>::widen(char) const/7122 to virtual std::ctype<char>::char_type std::ctype<char>::do_widen(char) const/1410
src/main.cpp:100:27: note: loop with 8 iterations completely unrolled (header execution count 118111597)
g++ -o exe/test.exe obj/main.o -O3 -std=c++11 -march=native -ftree-vectorize -fopt-info -Iinclude   -lm
Temps scalaire : 1.11319e-09
Temps vectorielle : 2.14818e-10
```

### Gaussian Filter

Apply `f` to element of the input vector where $f(x_j) = x_{j-1} + 2x_{j} + x_{j+1}$

The first and last element of the vector are treated as special cases.

**Scalar** :

```C
S[0] = (2 * A[0] + A[1]) / 4;
for (unsigned long j = 1; j < size-1; j++) {
    S[j] = (A[j-1] + 2*A[j] + A[j+1]) / 4;
}
S[size-1] = (2 * A[size-1] + A[size-2]) / 4;
```

**Vectorial** :

```C
__m256 two = _mm256_set1_ps(2.0);
__m256 four = _mm256_set1_ps(4.0);

__m256 a0 = _mm256_loadu_ps(A);
__m256 b0 = _mm256_loadu_ps(A+1);
__m256 c0 = _mm256_set_ps(A[6], A[5], A[4], A[3], A[2], A[1], A[0], 0);

// ( c0 + 2a0 + b0 ) / 4
__m256 gauss0 = _mm256_div_ps(_mm256_add_ps(_mm256_add_ps(b0, _mm256_mul_ps(two, a0)), c0), four);
_mm256_storeu_ps(S, gauss0);

for (unsigned long j = 8; j < size; j+=8) {
    __m256 a = _mm256_loadu_ps(A+j);
    __m256 b = _mm256_loadu_ps(A+j+1);
    __m256 c = _mm256_loadu_ps(A+j-1);

    // ( c + 2a + b ) / 4
    __m256 gauss = _mm256_div_ps(_mm256_add_ps(_mm256_add_ps(b, _mm256_mul_ps(two, a)), c), four);
    _mm256_storeu_ps(S+j, gauss);
}

__m256 af = _mm256_loadu_ps(A+size-8);
__m256 bf = _mm256_set_ps(0, A[size-1], A[size-2], A[size-3], A[size-4], A[size-5], A[size-6], A[size-7]);
__m256 cf = _mm256_loadu_ps(A+size-9);

__m256 gaussf = _mm256_div_ps(_mm256_add_ps(_mm256_add_ps(bf, _mm256_mul_ps(two, af)), cf), four);
_mm256_storeu_ps(S+size-8, gaussf);

```

- sans vectorisation automatique (optimization 3)

```
g++ -O2 -std=c++11 -march=native -fno-tree-vectorize -mavx -Iinclude -c src/main.cpp -o obj/main.o
g++ -o exe/test.exe obj/main.o -O2 -std=c++11 -march=native -fno-tree-vectorize -mavx -Iinclude   -lm
size : 1048576
temps scalaire 3.77563e-09
temps vectoriel 6.62101e-10
```

- avec vectorisation verbose

```
g++ -O3 -std=c++11 -march=native -ftree-vectorize -fopt-info -Iinclude -c src/main.cpp -o obj/main.o
/usr/local/include/c++/8.2.0/bits/locale_facets.h:877:23: note: speculatively devirtualizing call in std::ctype<char>::char_type std::ctype<char>::widen(char) const/7122 to virtual std::ctype<char>::char_type std::ctype<char>::do_widen(char) const/1410
src/main.cpp:50:37: note: loop vectorized
src/main.cpp:50:37: note: loop with 6 iterations completely unrolled (header execution count 15068648)
src/main.cpp:15:5: note: loop with 6 iterations completely unrolled (header execution count 14943801)
g++ -o exe/test.exe obj/main.o -O3 -std=c++11 -march=native -ftree-vectorize -fopt-info -Iinclude   -lm
size : 1048576
temps scalaire 1.73349e-09
temps vectoriel 6.62922e-10
```

### Absolute Value

**Scalar** :

```C++
for (unsigned long j = 0; j < size-1; j++) {
    S[j] = fabsf(A[j]);
}
```

**Vectorial** :

```C++
for (unsigned long j = 0; j < size; j+=8) {
            __m256 a = _mm256_loadu_ps(A+j);
            __m256 b = _mm256_sub_ps(zero, a);

            _mm256_storeu_ps(S_simd+j, _mm256_max_ps(a, b));
}
```

- sans vectorisation automatique (optimization 3)

```
g++ -O2 -std=c++11 -march=native -fno-tree-vectorize  -Iinclude -c src/main.cpp -o obj/main.o
g++ -o exe/test.exe obj/main.o -O2 -std=c++11 -march=native -fno-tree-vectorize  -Iinclude   -lm
size : 1048576
temps scalaire 7.64201e-10
temps vectoriel 6.57077e-10
```

- avec vectorisation verbose

```
g++ -O3 -std=c++11 -march=native -ftree-vectorize -fopt-info -Iinclude -c src/main.cpp -o obj/main.o
/usr/local/include/c++/8.2.0/bits/locale_facets.h:877:23: note: speculatively devirtualizing call in std::ctype<char>::char_type std::ctype<char>::widen(char) const/7123 to virtual std::ctype<char>::char_type std::ctype<char>::do_widen(char) const/1410
src/main.cpp:51:37: note: loop vectorized
src/main.cpp:51:37: note: loop with 6 iterations completely unrolled (header execution count 46365075)
g++ -o exe/test.exe obj/main.o -O3 -std=c++11 -march=native -ftree-vectorize -fopt-info -Iinclude   -lm
size : 1048576
temps scalaire 6.3992e-10
temps vectoriel 6.56035e-10
```

**Results:**
We would expect that the vectorial calculations will take less time than the scalar calculations. However, this is not always the case. There is often a lot of overhead to transform the scalars to a vectorized format. Perhaps with vectors of a bigger size, the difference in time would be more apparent ?
