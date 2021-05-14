__kernel void horizontal_step(__global const int *H,
                          __global const float *codeword,
                          const int N,
                          __global float *E)
    {
        int id = get_global_id(0);

        if(H[id] == 1){
            int temp_row = id / N;
            int temp_col = id % N;
            float min = 100000;
            int sign_product = 1;
            for(int i = 0; i < temp_col; i++){
                if(fabs(codeword[i] * H[i + temp_row * N]) < min && H[i + temp_row * N] != 0)
                    min = fabs(codeword[i] * H[i + temp_row * N]);
                if(codeword[i] < 0 && H[i + temp_row * N] != 0) 
                        sign_product *= -1;
                
            }
            for(int i = temp_col + 1; i < N; i++){
                if(fabs(codeword[i] * H[i + temp_row * N]) < min && H[i + temp_row * N] != 0)
                    min = fabs(codeword[i] * H[i + temp_row * N]);
                if(codeword[i] < 0 && H[i + temp_row * N] != 0) 
                        sign_product *= -1;
                
            }
            E[id] = min * sign_product;
        }
        else {
            E[id] = 0; }
    }