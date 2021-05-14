__kernel void fromBitToLLR(
                        __global float *codeword)

        {
            int id = get_global_id(0);
            //printf("\nhi i am inside gpu of thread %d ", codeword[id]);
            const float mistake_pr = 0.2;
            if(codeword[id] == 0){
                codeword[id] = log((1 - mistake_pr) / mistake_pr);
            }
            else {
                codeword[id] = log(mistake_pr / (1 - mistake_pr));
            }
            
        }