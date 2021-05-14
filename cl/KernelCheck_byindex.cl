__kernel void check(__global const int *H,
                          __global const int *codeword,
                          const int N,
                          const int b, 
                          __global int *syndrom)
{
    int id = get_global_id(0);

    int acc = 0;
    for(int i = 0; i < b; i++){
        acc = codeword[H[id * b + i] % N] ^ acc;
    }
    
    syndrom[id] = acc;
}