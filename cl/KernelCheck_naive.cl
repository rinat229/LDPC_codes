__kernel void check(__global const int *H,
                          __global const int *codeword,
                          const int N,
                          __global int *syndrom)
        {
            int id = get_global_id(0);

            int acc = 0;
            for(int i = 0; i < N; i++){
                acc = (codeword[i] * H[id * N + i]) ^ acc;
            }
            
            syndrom[id] = acc;
        }