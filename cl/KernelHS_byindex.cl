__kernel void horizontal_step(__global const int *H,
                          __global const float *codeword,
                          const int N,
                          const int b,
                          const int l,
                          __global float *E)
        {
            int id = get_global_id(0);
            int temp_row = H[id] / N;
            float min = 100000;
            bool sign_product = true;
            int temp_one_col = H[id] % N / l;
            for(int i = 0; i < temp_one_col; i++){
                if(fabs(codeword[H[i + temp_row * b] % N]) < min){
                    min = fabs(codeword[H[i + temp_row * b] % N]);
                }
                sign_product ^= codeword[H[i + temp_row * b] % N] < 0;
            }
            for(int i = temp_one_col + 1; i < b; i++){
                if(fabs(codeword[H[i + temp_row * b] % N]) < min){
                    min = fabs(codeword[H[i + temp_row * b] % N]);                    
                }
                sign_product ^= codeword[H[i + temp_row * b] % N] < 0;
            }
            E[id] = min * (2 * sign_product - 1);
        }