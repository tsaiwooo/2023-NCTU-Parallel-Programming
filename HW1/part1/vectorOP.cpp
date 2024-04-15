#include "PPintrin.h"

float sub_sum(float *,int ,int );

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //

  __pp_vec_float dst_vec;
  __pp_vec_int exp_vec;
  __pp_vec_float result;
  __pp_vec_int ones=_pp_vset_int(1);
  __pp_vec_int zeros_vec = _pp_vset_int(0);
  __pp_vec_float limit = _pp_vset_float(9.999999f);

  __pp_mask maskAll, maskIsone, maskIszero,mask_vgt;


  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    int non_zeros = 0;
    // init to ones
    //if(cur_len >= VECTOR_WIDTH){
    if((N-i)>=VECTOR_WIDTH){
      maskAll = _pp_init_ones(VECTOR_WIDTH);
    }
    //else if(cur_len<VECTOR_WIDTH){
    else if((N-i)<VECTOR_WIDTH){
      maskAll = _pp_init_ones(N-i);
    }

    //init to zero
    maskIsone = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(dst_vec, values + i, maskAll); // dst = values[i];
    //load original value to result
    _pp_vmove_float(result,dst_vec,maskAll);
    _pp_vload_int(exp_vec, exponents + i,maskAll);// load exponents;

    //set mask which exponent = 0
    _pp_veq_int(maskIszero,exp_vec,zeros_vec,maskAll);
    //if exponent==0 then set result 1
    _pp_vset_float(result,1,maskIszero);
    //else exponent >= 1 then calculate
    _pp_vgt_int(maskIsone,exp_vec,zeros_vec,maskAll);

    //sub 1 in exponent
    _pp_vsub_int(exp_vec,exp_vec,ones,maskIsone);
    //set mask when exponent > 1
    _pp_vgt_int(maskIsone,exp_vec,zeros_vec,maskAll);

    //count 1
    non_zeros = _pp_cntbits(maskIsone);
    while(non_zeros){
      //result * a
      _pp_vmult_float(result,result,dst_vec,maskIsone);
      //exponent--
      _pp_vsub_int(exp_vec,exp_vec,ones,maskIsone);

      //count how many 1 in mask
      _pp_vgt_int(maskIsone,exp_vec,zeros_vec,maskIsone);
      non_zeros = _pp_cntbits(maskIsone);
    }

  
    //if>9.999999f then set to 9.999999f
    _pp_vgt_float(mask_vgt,result,limit,maskAll);
    _pp_vset_float(result,9.999999f,mask_vgt);


    //store ans back
    _pp_vstore_float(output + i, result, maskAll);

  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  __pp_vec_float dst_vec,result;
  __pp_mask maskAll, maskIsone, maskIszero;
  __pp_mask one_element;
  one_element = _pp_init_ones(1);

  int len = VECTOR_WIDTH;
  float tmp_array[VECTOR_WIDTH+3];
  for(int i=0;i<VECTOR_WIDTH+3;i++){
    tmp_array[i] = 0;
  }
  float res_one[1],ans;
  int now_ptr = 0;
  

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    tmp_array[now_ptr] = sub_sum(values,VECTOR_WIDTH,i);
    now_ptr++;
    //start to calculate
    while(now_ptr>=(VECTOR_WIDTH)){
        now_ptr -= VECTOR_WIDTH;
        
        tmp_array[now_ptr] = sub_sum(tmp_array,VECTOR_WIDTH,now_ptr);
        for(int j=1;j<=VECTOR_WIDTH;j++){
          tmp_array[j] = 0;
        }
        now_ptr++;
    }
  }
  if(now_ptr>1){
    return sub_sum(tmp_array,now_ptr,0);
  }else{
    return tmp_array[0];
  }



  // return 0.0;
}

float sub_sum(float *values,int len,int start)
{
  __pp_vec_float dst_vec,result;
  __pp_mask maskAll, maskIsone, maskIszero;

  __pp_mask one_element;
  one_element = _pp_init_ones(1);
  maskIsone = _pp_init_ones();
  _pp_vset_float(dst_vec,0,maskIsone);

  int tmp_len = VECTOR_WIDTH;
  maskAll = _pp_init_ones();
  while(tmp_len!=1){
    //open mask of tmp_len
    // maskAll = _pp_init_ones(tmp_len);
    //load into vec
    _pp_vload_float(dst_vec,values+start,maskAll);
    //hadd
    _pp_hadd_float(result,dst_vec);
    //interleaving
    _pp_interleave_float(result,result);
    //store to array
    // _pp_vstore_float(values+start,result,maskAll);
    // maskAll = _pp_init_ones(len);
    _pp_vstore_float(values+start,result,maskAll);
    //stop condition
    tmp_len = tmp_len/2;
  }
  return values[start];

}
