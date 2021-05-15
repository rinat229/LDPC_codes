#include <random>
#include <algorithm>
#include <iostream>
#include <vector>
//#include "matrix_generate.h"

/*void check_matrix_generate(const int a, const int b, const int l, std::vector <int> &H){
    std::vector<int> repeat (a * b * b);
    for(int i = 0; i < b; i++){
        for(int k = 0; k < b * a; k++){
            if(k >= i * a && k < a * (i + 1)){
                repeat[i * a * b + k] = 1;
            }
            else{
                repeat[i * a * b + k] = 0;
            }
        }
    }

    std::copy(repeat.begin(), repeat.end(), H.begin());
    std::random_device rd;
    std::mt19937 g(rd());
    std::vector<int> for_perm(a * b);
    std::iota(for_perm.begin(), for_perm.end(), 0);

    for(int i = 0; i < l; i++){
        std::shuffle(for_perm.begin(), for_perm.end(), g);
        for(int j = 0; j < b; j++){
            for(int k = 0; k < b * a; k++){
                H[i * a * b * b + j * a * b + k] = H[j * a * b + for_perm[k]];
            }
        }
    }
}*/

void check_matrix_generate(const int a, const int b, const int l, std::vector<int> &the_matrix, std::vector<int> &matrix_of_index){
    std::random_device rd;
    std::mt19937 g(rd());
    std::vector<int> offsetvector(a * b);
    std::iota(offsetvector.begin(), offsetvector.end(), 0);
    std::shuffle(offsetvector.begin(), offsetvector.end(), g);
    int index = 0;   
    for(int i = 0; i < a; i++){
        for(int k = 0; k < b; k++){  
            for(int j = 0; j < l; j++){
                the_matrix[l * (b * (i * l + j) + k) + (j + offsetvector[i * b + k]) % l] = 1;
                matrix_of_index[index] = l * (b * (i * l + j) + k) + (j + offsetvector[i * b + k]) % l;
                index++;
            }
        }
    }
}

void rowsswap(const int row1, const int row2, const int col_num, std::vector<int> &H){
    for(int element = 0; element < col_num; element++){
        std::swap(H[row1 * col_num + element], H[row2 * col_num + element]);
    }
}
void rows_sum(const int row1, const int row2, const int col_num, std::vector<int> &the_matrix){
    for(int element = 0; element < col_num; element++){
        the_matrix[row1 * col_num + element] = the_matrix[row1 * col_num + element] ^ the_matrix[row2 * col_num + element];
    }
}

void row_erase(const int row1, const int col_num, std::vector<int> &the_matrix){
    the_matrix.erase(the_matrix.begin() + row1 * col_num, the_matrix.begin() + row1 * (col_num + 1));
}
void gen_matrix_without_upper(int &row_num, const int col_num, std::vector<int> &the_matrix){
    const int l = col_num - row_num;
    
    for (int pivot = 0; pivot < row_num; pivot++){
        for(int j = pivot; j < row_num; j++){
            if(the_matrix[j * col_num + pivot + l] == 1){
                rowsswap(pivot, j, col_num, the_matrix);
                break;
            }
        }
        for(int under_pivot = pivot + 1; under_pivot < row_num; under_pivot++){
            if(the_matrix[under_pivot * col_num + pivot + l] == 1){
                rows_sum(under_pivot, pivot, col_num, the_matrix);
            }
        }
    }
    
}
void gen_matrix_with_upper(int row_num, int col_num, std::vector<int> &the_matrix){
    const int l = col_num - row_num;
    for (int pivot = 1; pivot < row_num; pivot++){
        for(int upper_pivot = pivot - 1; upper_pivot >= 0; upper_pivot--){
            if(the_matrix[upper_pivot * col_num + pivot + l] == 1){
                rows_sum(upper_pivot, pivot, col_num, the_matrix);
            }
        }
    }
}
void gen_matrix(int &row_num, int col_num, std::vector<int> &the_matrix){
    gen_matrix_without_upper(row_num, col_num, the_matrix);
    for(int l = row_num - 1; l > 0; l--){
            if(the_matrix[l * col_num + col_num - row_num + l] == 0){
                row_erase(l, col_num, the_matrix);
                row_num--;
            }
            else
                break;
    }
    gen_matrix_without_upper(row_num, col_num, the_matrix);
    gen_matrix_with_upper(row_num, col_num, the_matrix);
}
template<typename T>
void codeword_generate(int row_num, int col_num, std::vector<T> &codeword, std::vector<int> &check_matrix){

    codeword[0] = 1;
    const int diff = col_num - row_num;
    for(int i = 1; i < diff; i++){
        codeword[i] = 0;
    }
    for(int i = 0; i < row_num; i++){
        codeword[i + diff] = check_matrix[i * col_num];
    }
}
int main(){
    int a = 3 , b = 4, l = 5;
    int old_row_num = l * a;
    int row_num = l * a, col_num = l * b;
    std::vector<int> H_by_index(a * b * l);
    std::vector<int> codeword(col_num);
    {   
        std::cout << "row number - " << row_num << std::endl; 
        std::vector<int> H(row_num * col_num);
        check_matrix_generate(a, b, l, H, H_by_index);

        for(int i = 0; i < row_num; i ++){
            for(int k = 0; k < col_num; k++){
                std::cout << H[i * col_num + k] << ' ';
            }
            std::cout << std::endl;
        }

        std::cout << std::endl << std::endl << std::endl;
        gen_matrix(row_num, col_num, H);
        for(int i = 0; i < row_num; i ++){
            for(int k = 0; k < col_num; k++){
                std::cout << H[i * col_num + k] << ' ';
            }
            std::cout << std::endl;
        }
        codeword_generate(row_num, col_num, codeword, H);
        std::cout << "row number - " << row_num << std::endl; 
        for(auto &a : codeword)
            std::cout << a << ' ';
    }
    std::cout << -1 - (-1/5) * 5;
    return 0;
}
