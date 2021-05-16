#include "Decoder.hpp"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

int a = 2, b = 3, l = 10;
// int main() {
//     DecoderMinSumByIndex decoder(2, 3, 10);
//     for (auto item : decoder.codeword)
//       std::cout << item << ' ';
//     std::cout << '\n';
//     mistake_generate(decoder.codeword);
//     for (auto item : decoder.codeword)
//       std::cout << item << ' ';
//     std::cout << '\n';
//     decoder.run();
//     for (auto item : decoder.codeword)
//       std::cout << item << ' ';
//     std::cout << '\n';
// }
TEST_CASE("zero codeword"){
  DecoderMinSumByIndex decoder(a, b, l);
  std::vector<float> zero_codeword(b * l);
  decoder.codeword = zero_codeword;
  decoder.run();
  REQUIRE(decoder.codeword == zero_codeword);
}


TEST_CASE("zero codeword with errors"){
  DecoderMinSumByIndex decoder(a, b, l);
  std::vector<float> zero_codeword(b * l);
  decoder.codeword = zero_codeword;
  srand(time(NULL));
  decoder.codeword[rand() % (b * l)] = 1;
  decoder.codeword[rand() % (b * l)] = 1;
  decoder.run();
  CHECK(decoder.codeword == zero_codeword);
}

TEST_CASE("some codeword without mistakes"){
  DecoderMinSumByIndex decoder(a, b, l);
  std::vector<float> codeword(b * l);
  codeword = decoder.codeword ;
  decoder.run();
  REQUIRE(decoder.codeword == codeword);
}

TEST_CASE("some codeword with mistakes"){
  DecoderMinSumByIndex decoder(a, b, l);
  std::vector<float> codeword(b * l);
  codeword = decoder.codeword;
  mistake_generate(decoder.codeword);
  decoder.run();
  CHECK(decoder.codeword == codeword);
}