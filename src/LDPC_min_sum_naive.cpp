// This decoder uses whole check matrix
// In other words, all steps run on every element even zeros


#define CL_TARGET_OPENCL_VERSION 120
#include <algorithm>
#include <iostream>
#include <vector>
#include <ctime>
#include <random>
#include <algorithm>
#include <chrono>

#include <boost/compute/core.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/utility/source.hpp>
#include <cmath>
#include "matrix_generate.h"
#include "Timer.h"
namespace compute = boost::compute;

const float mistake_pr = 0.001;


void mistake_generate(std::vector<float> &codeword){
    srand(time(0));
    for(auto &a : codeword){
        if((double)rand()/RAND_MAX < mistake_pr)
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
    
    // matrix parameters 
    // a = numbers of ones in a row
    // b = numbers of ones in a column 
    // l = length of identity matrix circulant
    // n, k = matrix shape
    int a = 2, b = 3, l = 1000; 

    size_t n = b * l, k = a * l; 
    int row_num = a * l, col_num = b * l;

    std::vector<float> codeword(n);
    std::vector<int> check_matrix(n * k);
    check_matrix_generate(a, b, l, check_matrix);
    {
        // generating codeword by eliminating a check matrix to gen matrix
        std::vector<int> check_matrix_copy(row_num * col_num);
        check_matrix_copy = check_matrix;
        gen_matrix(row_num, col_num, check_matrix_copy);
        codeword_generate(row_num, col_num, codeword, check_matrix_copy);
    }
    // making an error
    mistake_generate(codeword);
    // Matrix E uses for finding minimal element and a sign product of nonzero elements in a row
    std::vector<float> E (n * k);
    // syndrom is equal H^T * codeword
    // codeword belongs codeword's space if syndrom is a zero vector
    std::vector<int> syndrom (k);

    compute::vector<float> buffer_codeword (codeword.begin(), codeword.end(), queue);
    compute::vector<int> buffer_check_matrix (check_matrix.begin(), check_matrix.end(), queue);
    compute::vector<float> buffer_E (n * k, context);
    compute::vector<int> buffer_syndrom (k, context);
    
    compute::program program_fromBitToLLR = compute::program::build_with_source_file("KernelLLR.cl", context);
    compute::program program_horizontal_step = compute::program::build_with_source_file("KernelHS_naive.cl", context);
    compute::program program_vertical_step = compute::program::build_with_source_file("KernelVS_naive.cl", context); 
    compute::program program_check_step = compute::program::build_with_source_file("KernelCheck_naive.cl", context); 

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

    int iterations_number = 0;
    int MAXiterations = 100;
    bool exit = true;
    Timer timer;

    // computing a syndrom and checking on equality to a zero
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
    timer.Stop();
    compute::copy(buffer_codeword.begin(), buffer_codeword.end(), codeword.begin(), queue);

    std::cout << std::endl << "number of iterations - " << iterations_number << std::endl;
    return 0;
}
