#define CL_TARGET_OPENCL_VERSION 120
#include <algorithm>
#include <iostream>
#include <vector>

#include <boost/compute/core.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/utility/source.hpp>
#include <cmath>

namespace compute = boost::compute;

// * * * * 
//  horizontal step
// * * * *


const float mistake_pr = 0.3;

compute::program make_HS_program(const compute::context& context)
{   
    const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
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
    );
    
    return compute::program::build_with_source(source, context);
}

// * * * * 
//  syndrom check
// * * * * 

compute::program make_check_program(const compute::context& context)
{
    const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
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
    );
    
    return compute::program::build_with_source(source, context);
}

// * * * * 
//  vertical step
// * * * *

compute::program make_VS_program(const compute::context& context)
{
    const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
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
    );
    
    return compute::program::build_with_source(source, context);
}

// * * * * 
//  from bit to LLR
// * * * *

compute::program make_fromBitToLLR_program(const compute::context& context)
{
    const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
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
    );
     
    return compute::program::build_with_source(source, context);
}
void mistake_generate(std::vector<float> &codeword){
    for(auto &a : codeword){
        if((double)(rand())/RAND_MAX < mistake_pr)
            if(a == 0)
                a = 1;
            else 
                a = 0;
    }
}

int main()
{
    compute::device gpu = compute::system::default_device();
    compute::context context(gpu);
    compute::command_queue queue(context, gpu);
    
    size_t n = 6, k = 4;

    //check matrix = [1  1  0  1  0  0]
    //               [0  1  1  0  1  0]
    //               [1  0  0  0  1  1]
    //               [0  0  1  1  0  1]
    std::vector<float> codeword {1, 0, 0, 0, 1, 1};
    std::cout << "right codeword" << std::endl;
    for(auto &a : codeword)
        std::cout << a << " ";
    std::cout << std::endl;
    mistake_generate(codeword);
    std::cout << "codeword after transmission on BSC" << std::endl;
    for(auto &a : codeword)
        std::cout << a << " ";
    std::cout << std::endl;
    std::vector<int> check_matrix {1, 1, 0, 1, 0, 0, 
                                   0, 1, 1, 0, 1, 0, 
                                   1, 0, 0, 0, 1, 1, 
                                   0, 0, 1, 1, 0, 1};
    std::vector<float> E (n * k);
    std::vector<int> syndrom (k);

    compute::vector<float> buffer_codeword (codeword.begin(), codeword.end(), queue);
    compute::vector<int> buffer_check_matrix (check_matrix.begin(), check_matrix.end(), queue);
    compute::vector<float> buffer_E (n * k, context);
    compute::vector<int> buffer_syndrom (k, context);
    
    compute::program program_fromBitToLLR = make_fromBitToLLR_program(context);
    compute::program program_horizontal_step = make_HS_program(context);
    compute::program program_vertical_step = make_VS_program(context); 
    compute::program program_check_step = make_check_program(context); 

    compute::kernel kernel_LLR(program_fromBitToLLR, "fromBitToLLR");
    compute::kernel kernel_HS(program_horizontal_step, "horizontal_step");
    compute::kernel kernel_VS(program_vertical_step, "vertical_step");
    compute::kernel kernel_check(program_check_step, "check");

    kernel_LLR.set_arg(0, buffer_codeword.get_buffer());

    kernel_HS.set_arg(0, buffer_check_matrix.get_buffer());
    kernel_HS.set_arg(1, buffer_codeword.get_buffer());
    kernel_HS.set_arg(2, (int)n);  
    kernel_HS.set_arg(3, buffer_E.get_buffer());

    kernel_VS.set_arg(0, buffer_codeword.get_buffer());
    kernel_VS.set_arg(1, (int)n);
    kernel_VS.set_arg(2, (int)k);  
    kernel_VS.set_arg(3, buffer_E.get_buffer());

    kernel_check.set_arg(0, buffer_check_matrix.get_buffer());
    kernel_check.set_arg(1, buffer_codeword.get_buffer());
    kernel_check.set_arg(2, (int)n);    
    kernel_check.set_arg(3, buffer_syndrom.get_buffer());

    int iterations_number = 1;
    int MAXiterations = 50;
    bool exit = true;

    for(iterations_number; iterations_number < MAXiterations; iterations_number++){
        queue.enqueue_1d_range_kernel(kernel_check, 0, k, 0);
        compute::copy(buffer_syndrom.begin(), buffer_syndrom.end(), syndrom.begin(), queue);    
        for(auto &a:syndrom){
            if(a != 0){
                exit = false;
                break;
            }
            exit = true;
        }             
        if(!exit){          
            queue.enqueue_1d_range_kernel(kernel_LLR, 0, n, 0);
            queue.enqueue_1d_range_kernel(kernel_HS, 0, n * k, 0);
            queue.enqueue_1d_range_kernel(kernel_VS, 0, n, 0);
        }
        else break;  
    }
    compute::copy(buffer_codeword.begin(), buffer_codeword.end(), codeword.begin(), queue);
    std::cout << "codeword after MinSum" << std::endl;    
    for(auto &a : codeword)
        std::cout << a << " ";
    std::cout << std::endl << "number of iterations - " << iterations_number << std::endl;
    return 0;
}
