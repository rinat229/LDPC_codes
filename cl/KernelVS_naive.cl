__kernel void vertical_step(
                          __global float *codeword,
                          const int N,
                          const int K,
                          __global float *E)
        {
            int id = get_global_id(0);
            float acc = 0;

            for(int i = 0; i < K; i++){
                acc += E[id + i * N];
            }

            if (codeword[id] + acc > 0)
                codeword[id] = 0;
            else 
                codeword[id] = 1;
        }
        