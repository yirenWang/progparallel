How to use GPU 

1. cuda malloc the variables that you need in the gpu version 
```C
cudaMalloc(&d_Source, height*width*sizeof(u_char));      
```

2. Copy over the data from host to device    
```
cudaMemcpy(d_Source, Source[0], height*width*sizeof(u_char), cudaMemcpyHostToDevice);
```

3. You can now call the gpu function that executes the code in the gpu
This function is called within a thread that is contained within a block. 
We have access to `blockIdx`, `blockDim` and `threadIdx`. 

To have a one on one 