#define CL_TARGET_OPENCL_VERSION 120
#include <algorithm>
#include <iostream>
#include <vector>

#include <boost/compute/core.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/utility/source.hpp>

namespace compute = boost::compute;

// * * * * 
//  vertical step
// * * * * 

compute::program make_HS_program(const compute::context& context)
{
    const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
        __kernel void horizontal_step(__global const int *H,
                          __global const int *codeword,
                          const int N,
                          __global int *E)
        {
            int id = get_global_id(0);
            int temp_row = id / N;
            //printf("\nhi i am inside gpu of thread %d ", id);
            //check

            if(H[id] == 1){
               int xor_s = 0;
               for(int i = 0; i < N; i++)
                    xor_s = ((codeword[i] * H[i + temp_row * N]) ^ xor_s);
            E[id] = (xor_s ^ (codeword[id % N] * H[id]));}
            
            else {
                E[id] = -2; }
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
            //printf("\nhi i am inside gpu of thread %d ", id);
            int acc = 0;
            for(int i = 0; i < N; i++){
                acc = (codeword[i] * H[id * N + i]) ^ acc;
            }
            //printf("\nhi i am inside gpu of thread %d ", acc);
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
                          __global int *codeword,
                          const int N,
                          const int K,
                          __global int *E)
        {
            int id = get_global_id(0);
            int temp_col = id / K;
            //printf("\nhi i am inside gpu of thread %d ", id);

            int zero_count = 0;
            int one_count = 0;
            for (int i = 0; i < K; i++){
                if(E[temp_col + i * N] == 1){
                    one_count += 1;
                }   
                else if (E[temp_col + i * N] == 0){
                    zero_count += 1; }
            }
            if (codeword[temp_col] == 1)
                if(zero_count > one_count)
                    codeword[temp_col] = 0;
            else
                if(one_count > zero_count)
                    codeword[temp_col] = 1;
        }
    );
    
    return compute::program::build_with_source(source, context);
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
    std::vector<int> codeword { 1, 0, 1, 0, 1, 1};
    std::vector<int> check_matrix {1, 1, 0, 1, 0, 0, 
                                   0, 1, 1, 0, 1, 0, 
                                   1, 0, 0, 0, 1, 1, 
                                   0, 0, 1, 1, 0, 1};
    std::vector<int> E (n * k);
    std::vector<int> syndrom (k);

    compute::vector<int> buffer_codeword (codeword.begin(), codeword.end(), queue);
    compute::vector<int> buffer_check_matrix (check_matrix.begin(), check_matrix.end(), queue);
    compute::vector<int> buffer_E (n * k, context);
    compute::vector<int> buffer_syndrom (k, context);

    compute::program program_horizontal_step = make_HS_program(context);
    compute::program program_vertical_step = make_VS_program(context); 
    compute::program program_check_step = make_check_program(context); 
    
    compute::kernel kernel_HS(program_horizontal_step, "horizontal_step");
    compute::kernel kernel_VS(program_vertical_step, "vertical_step");
    compute::kernel kernel_check(program_check_step, "check");

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
                std::cout<<"break";
                break;
            }
            exit = true;
        }             
        if(!exit){          
            queue.enqueue_1d_range_kernel(kernel_HS, 0, n * k, 0);
            queue.enqueue_1d_range_kernel(kernel_VS, 0, n * k, 0);
        }
        else break;  
    }
    compute::copy(buffer_codeword.begin(), buffer_codeword.end(), codeword.begin(), queue);

    for(auto &a : codeword)
        std::cout << a << " ";
    std::cout << std::endl << iterations_number;
    return 0;
}
