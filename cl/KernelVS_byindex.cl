__kernel void vertical_step(
                          __global float *codeword,
                          const int N,
                          const int a,
                          const int b,
                          const int l,
                          __global float *E,
                          __global int *H)
{
    int id = get_global_id(0);
    float acc = 0;
    int temp_one_col = (id % N) / l;

    for(int i = 0; i < a; i++){
        int idx = id % l - ( H[temp_one_col + i * b * l] % N ) % l;
        if (idx < 0){
            idx += l;
        }
        acc += E[i * b * l + (idx % l) * b + temp_one_col];
    }

    if (codeword[id] + acc > 0)
        codeword[id] = 0;
    else 
        codeword[id] = 1;
}