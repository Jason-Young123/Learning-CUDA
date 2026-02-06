#include <vector>
#include <musa_fp16.h>

#include "../tester/utils.h"

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */

template <typename T>
__device__ T warp_reduce_sum(T val){
#pragma unroll//短循环自动展开,省去分支预测,提升效率
    for(int offset = 16; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}


template <typename T>
__global__ void trace_calc(T* d_trace, const T* d_diag, size_t n){
  __shared__ T smem[32];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  //通过两级归约(warp内, block内/warp间)完成所有元素相加
  T sum = T(0);
  for(size_t i = idx; i < n; i += stride){
    sum += d_diag[i];
  }

  //一级归约,每个warp内完成规约
  T warp_sum = warp_reduce_sum(sum);

  //准备二级规约,将每个warp内lane[0]的值拷贝到smem中
  if((tid % 32) == 0){
      smem[tid / 32] = warp_sum;
  }
  __syncthreads();//等待拷贝至smem操作完成

  //需保证一级归约后每个block内的线程不超过32(即block_dim不超过32*32 = 1024)
  if(tid < 32){
      //准备二级规约,多余线程补0
      T block_sum = (tid < (blockDim.x + 31)/32) ? smem[tid] : T(0);
      //二级归约
      block_sum = warp_reduce_sum(block_sum);
      if(tid == 0 && block_sum != T(0)){//原子操作次数 = block数量
        atomicAdd(d_trace, block_sum);
      }
  }  
  return;
}


//提取对角元素
template <typename T>
std::vector<T> extract_diag(const std::vector<T> & h_input, size_t rows, size_t cols){
  size_t n = std::min(rows, cols);
  std::vector<T> diag(n);
  for(size_t i = 0; i < n; ++i){
    diag[i] = h_input[i * cols + i];
  }
  return diag;
}




template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  //step-0: basic check
  //printf("=== MUSA Device Properties ===\n");
  //musaDeviceProp prop; 
  //int device_id = 0; 
  //musaGetDeviceProperties(&prop, device_id);
  
  // 静态property
  //printf("Device Name: %s\n", prop.name);
  //printf("  - Max Threads per Block: %d\n", prop.maxThreadsPerBlock);//1024
  //printf("  - Max Block Dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);//(1024, 1024, 1024)
  //printf("  - Max Grid Dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);//(2147483647, 2147483647, 2147483647)
  //printf("  - Warp Size: %d\n", prop.warpSize);//32 
  //printf("  - Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);//192KB
  //printf("  - Number of Multiprocessors: %d\n", prop.multiProcessorCount);//56
  
  
  if(!std::min(rows, cols)){
    //std::cerr << "Matrix Shape Invalid" << std::endl;
    return T(0);
  }

  //step-1: 提取对角元
  std::vector<T> h_diag = extract_diag<T>(h_input, rows, cols);

  //step-2: 初始化,分配device端空间
  const size_t size_bytes = h_diag.size() * sizeof(T);
  T *d_diag, *d_trace;//device端只支持裸指针
  RUNTIME_CHECK(musaMalloc(&d_diag, size_bytes));
  RUNTIME_CHECK(musaMalloc(&d_trace, sizeof(T)));

  //step-3: 拷贝数据from host to device
  RUNTIME_CHECK(musaMemcpy(d_diag, h_diag.data(), size_bytes, musaMemcpyHostToDevice));
  RUNTIME_CHECK(musaMemset(d_trace, 0, sizeof(T)));

  //step-4: device端计算
  int block_dim = 1024;
  int grid_dim = std::min((h_diag.size() + block_dim - 1)/block_dim, size_t(8));//设置上限
  trace_calc<T><<<grid_dim, block_dim>>>(d_trace, d_diag, h_diag.size());//调用device端函数进行trace计算
  //注意核函数返回类型只能为void

  //step-5: 拷贝数据from device to host
  T h_trace = T(0);
  RUNTIME_CHECK(musaMemcpy(&h_trace, d_trace, sizeof(T), musaMemcpyDeviceToHost));

  //step5: free memory
  RUNTIME_CHECK(musaFree(d_diag));
  RUNTIME_CHECK(musaFree(d_trace));

  //printf("the result is %f\n", float(h_trace));
  return h_trace;
}






/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */

//moore平台只支持c++11, 需去除constexpr
/*template <typename T>
__device__ T myexp(T x) {
    if constexpr(std::is_same<T, __half>::value) {
        float fx = __half2float(x);
        float result = expf(fx);
        return __float2half(result);
    }
    else if constexpr(std::is_same<T, float>::value) {
        return expf(x);  // expf返回float
    }
    else if constexpr(std::is_same<T, double>::value) {
        return exp(x);   // exp返回double
    }
    else{//other types
      return T(0);
    }
}*/


template <typename T>
__device__ T warp_reduce_max(T val){
#pragma unroll//短循环自动展开,省去分支预测,提升效率
    for(int offset = 16; offset > 0; offset >>= 1){
        T tmp = __shfl_down_sync(0xffffffff, val, offset);
        val = (val > tmp) ? val : tmp;
    }
    return val;
}


//对应flash attention v2原文算法, block采用三维布局
template <typename T>
__global__ void kernel_flashAttention(int batch_size, int target_seq_len, int src_seq_len, int q_heads, int kv_heads, int head_dim, bool is_causal, const T* Q, const T* K, const T* V, T* O){
  int tid_x = threadIdx.x;//横向,blockDim.x列
  int tid_y = threadIdx.y;//纵向,blockDim.y行
  int bid_x = blockIdx.x;//x方向,总数 = #q_heads
  int bid_y = blockIdx.y;//y方向,总数 = #batch
  int bid_z = blockIdx.z;//z方向,总数 = Tr
  const int p = q_heads / kv_heads;//计算比例系数
  const int Br = blockDim.y;//Q纵向每块大小, 默认为16 (RTX 5090)
  const int Bc = blockDim.x;//K/V纵向分块大小, 默认为16
  const int Tc = (src_seq_len + Bc - 1) / Bc;//对应原始论文中K/V纵向分块数Tc,其中Bc = 32

  //预计算常量
  const double scale_factor = 1.0 / sqrt(double(head_dim));//保留精度,采用double

  //定义一系列临时变量
  /*__shared__ double SP[Br][Bc];//复用S和P
  __shared__ double m_prev[Br], m_new[Br];
  __shared__ double l_prev[Br], l_new[Br];

  __shared__ float Q_sm[Br][64];
  __shared__ float K_T_sm[64][Bc];//transpose for K
  __shared__ float V_sm[Bc][64];
  __shared__ float O_sm[Br][64];*/

  extern __shared__ char shared_mem[];
  char* ptr = shared_mem;  
  //计算中间变量,包括S, P(复用为SP), m_prev, m_new, l_prev, l_new; 为保留精度, SP采用double
  double* SP = reinterpret_cast<double*>(ptr);    // double SP[Br][Bc]
  ptr += Br * Bc * sizeof(double);
  float* m_prev = reinterpret_cast<float*>(ptr);  // float m_prev[Br]
  ptr += Br * sizeof(float);
  float* m_new = reinterpret_cast<float*>(ptr);   // float m_new[Br] 
  ptr += Br * sizeof(float);
  float* l_prev = reinterpret_cast<float*>(ptr);  // float l_prev[Br]
  ptr += Br * sizeof(float);
  float* l_new = reinterpret_cast<float*>(ptr);   // float l_new[Br] 
  ptr += Br * sizeof(float);  

  //原始数据QKV和计算结果O; 全采用float
  float* Q_sm = reinterpret_cast<float*>(ptr);    // float Q_sm[Br][head_dim] 
  ptr += Br * head_dim * sizeof(float);  
  float* K_T_sm = reinterpret_cast<float*>(ptr);  // float K_T_sm[head_dim][Bc]
  ptr += head_dim * Bc * sizeof(float);
  float* V_sm = reinterpret_cast<float*>(ptr);    // float V_sm[Br][head_dim] 
  ptr += Bc * head_dim * sizeof(float);  
  float* O_sm = reinterpret_cast<float*>(ptr);    // float O_sm[Br][head_dim]

  //定义访问宏
  /*#define   SP_AT(y, x)       SP[y][x]
  #define   Q_sm_AT(y, x)     Q_sm[y][x]
  #define   K_T_sm_AT(y, x)   K_T_sm[y][x]
  #define   V_sm_AT(y, x)     V_sm[y][x]
  #define   O_sm_AT(y, x)     O_sm[y][x]*/

  #define   SP_AT(y, x)       SP[y * Bc + x]
  #define   Q_sm_AT(y, x)     Q_sm[y * head_dim + x]
  #define   K_T_sm_AT(y, x)   K_T_sm[y * Bc + x]
  #define   V_sm_AT(y, x)     V_sm[y * head_dim + x]
  #define   O_sm_AT(y, x)     O_sm[y * head_dim + x]


  /****************************preparation**************************/
  int bound_tid_y = ::min(Br, target_seq_len - Br * bid_z);

  //preparation-1: load Qi from GM to SM, and reset Oi to 0
  //Q[bid_y][Br * bid_z + tid_y][bid_x][*]
  for(int idx = tid_x; idx < head_dim; idx += blockDim.x){
    O_sm_AT(tid_y, idx) = 0.0;
    Q_sm_AT(tid_y, idx) = 0.0;
    if(tid_y < bound_tid_y){
      Q_sm_AT(tid_y, idx) = float(Q[((((bid_y * target_seq_len) + (Br * bid_z + tid_y)) * q_heads) + bid_x) * head_dim + idx]);
    }
  }
  __syncthreads();

  //preparation-2: reset m_prev to -INFINITY and l_prev to 0
  if(tid_x == 0){
    m_prev[tid_y] = -8192.0;
    l_prev[tid_y] = 0.0;
  }
  __syncthreads();
  /****************************end-of-preparation*************************/


  /****************************main-loop**************************/
  for(int j = 0; j < Tc; ++j){//对于每个K/V分块
    if(is_causal && bid_z < j){//early exit, 直接跳过
    //__syncthreads();
      continue;
    }

    SP_AT(tid_y, tid_x) = -8192.0;
    __syncthreads();
    int bound_tid_x = ::min(Bc, src_seq_len - Bc * j);
    bool is_compute = true;//optimization: 分支处理,加速branch-resolving
    if (is_causal) {
      if (bid_z < j) {
        is_compute = false;  // 早期退出情况
      } else if (bid_z == j) {
        is_compute = (tid_y >= tid_x);  // 对角线以上
      }
    }

    //step-1: load Ki, Vi from GM to SM, reset Oi to 0
    //K[bid_y][Bc * j + tid_y][bid_x / p][*]
    //V[bid_y][Bc * j + tid_y][bid_x / p][*]
    for(int idx = tid_x; idx < head_dim; idx += blockDim.x){
      K_T_sm_AT(idx, tid_y) = 0.0;
      V_sm_AT(tid_y, idx) = 0.0;
      if(tid_y < bound_tid_x){//注意这里是bound_tid_x
        K_T_sm_AT(idx, tid_y) = float(K[((((bid_y * src_seq_len) + (Bc * j + tid_y)) * kv_heads) + (bid_x / p)) * head_dim + idx]);
        V_sm_AT(tid_y, idx) = float(V[((((bid_y * src_seq_len) + (Bc * j + tid_y)) * kv_heads) + (bid_x / p)) * head_dim + idx]);
      }
    }
    __syncthreads();


    //step-2: S = Q @ K.T, point-wise
    if(tid_y < bound_tid_y && tid_x < bound_tid_x){//用于边缘不完整块
      float val0 = 0.0;//临时sum
      if(is_compute){
        for(int k = 0; k < head_dim; ++k){
          val0 += Q_sm_AT(tid_y, k) * K_T_sm_AT(k, tid_x);
        }
        SP_AT(tid_y, tid_x) = double(val0) * scale_factor;//必须用double,对精度影响最大的计算步骤
      }
    }
    __syncthreads();

    
    //step-3: m_new = max(m_prev, rowMax(S))
    float val1 = float(SP_AT(tid_y, tid_x));
    val1 = warp_reduce_max(val1);
    if(tid_x == 0 && tid_y < bound_tid_y){
      /*double val1 = SP_AT(tid_y, 0);//手动实现非并行求行最大值
      for(int h = 1; h < Bc; ++h){
        val1 = (val1 < SP_AT(tid_y, h)) ? SP_AT(tid_y, h) : val1;
      }*/
      m_new[tid_y] = (val1 > m_prev[tid_y]) ? val1 : m_prev[tid_y];
    }
    __syncthreads();

    //step-4: P = exp(S - m_new), point-wise
    if(tid_y < bound_tid_y && tid_x < bound_tid_x){
      if(is_compute){
        SP_AT(tid_y, tid_x) = exp(SP_AT(tid_y, tid_x) - double(m_new[tid_y]));
      }
      else{
        SP_AT(tid_y, tid_x) = 0.0;
      }
    }
    else{
      SP_AT(tid_y, tid_x) = 0.0;
    }
    
    __syncthreads();

    //step-5: l_new = exp(m_prev - m_new) * l_prev + rowSum(P)
    float val2 = float(SP_AT(tid_y, tid_x));
    val2 = warp_reduce_sum(val2);
    float exp_result = expf(m_prev[tid_y] - m_new[tid_y]);
    //float exp_result = expf(m_prev[tid_y] - m_new[tid_y]);
    if(tid_x == 0 && tid_y < bound_tid_y){
      /*double val2 = 0.0;//手动实现非并行求rowSum
      for(int h = 0; h < Bc; ++h){
        val2 += SP_AT(tid_y, h);
      }*/
      l_new[tid_y] = exp_result * l_prev[tid_y] + val2;
    }
    __syncthreads();


    //step-6: O = 1/(exp(m_prev - m_new)) * O + P @ V
    if(tid_x < bound_tid_x && tid_y < bound_tid_y){//32路并行计算Oi的每一行
      for(int u = tid_x; u < head_dim; u += blockDim.x){
        float val3 = 0.0;
        for(int w = 0; w < Bc; ++w){//val3 += P[tid_y][w] * V[bid_y][Bc * j + w][bid_x / p][u];
          val3 += float(SP_AT(tid_y, w)) * V_sm_AT(w, u);
        }
        O_sm_AT(tid_y, u) = O_sm_AT(tid_y, u) * exp_result + val3;
      }
    }
    __syncthreads();
      
    //step-7: m_prev <- m_new; l_prev <- l_new
    if (tid_x == 0 && tid_y < bound_tid_y) {//向量更新只使用第1列线程
      m_prev[tid_y] = m_new[tid_y];
      l_prev[tid_y] = l_new[tid_y];
    }
    __syncthreads();

  }
  /****************************end-of-main-loop**************************/

  /*****************************post-process****************************/
  //O(GM) = O/l_prev, aka O_sm /= l_prev and write Oi from SM to GM
  //O[bid_y][Br * bid_z + tid_y][bid_x][*]
  for(int idx = tid_x; idx < head_dim; idx += blockDim.x){
    if(tid_y < bound_tid_y){
      O[((((bid_y * target_seq_len) + (Br * bid_z + tid_y)) * q_heads) + bid_x) * head_dim + idx] = T(O_sm_AT(tid_y, idx) / float(l_prev[tid_y]));
    }
  }
  __syncthreads();
  /*****************************end-of-post-process****************************/

  //取消访问宏定义
  #undef   SP_AT
  #undef   Q_sm_AT
  #undef   K_T_sm_AT
  #undef   V_sm_AT
  #undef   O_sm_AT

}



template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  //step0: basic check

  //step1: 初始化,预留device端空间
  const size_t size_bytes_q = h_q.size() * sizeof(T);
  const size_t size_bytes_k = h_k.size() * sizeof(T);
  const size_t size_bytes_v = h_v.size() * sizeof(T);
  const size_t size_bytes_o = h_o.size() * sizeof(T);
  //const size_t size_bytes_lm = target_seq_len * query_heads * batch_size * sizeof(T);
  T *d_q, *d_k, *d_v, *d_o;//device端只支持裸指针
  RUNTIME_CHECK(musaMalloc(&d_q, size_bytes_q));
  RUNTIME_CHECK(musaMalloc(&d_k, size_bytes_k));
  RUNTIME_CHECK(musaMalloc(&d_v, size_bytes_v));
  RUNTIME_CHECK(musaMalloc(&d_o, size_bytes_o));
  //RUNTIME_CHECK(musaMalloc(&d_l, size_bytes_lm));//l向量,长度 = target_seq_len
  //RUNTIME_CHECK(musaMalloc(&d_m, size_bytes_lm));//m向量,长度 = target_seq_len

  //step2: 拷贝数据from host to device
  RUNTIME_CHECK(musaMemcpy(d_q, h_q.data(), size_bytes_q, musaMemcpyHostToDevice));
  RUNTIME_CHECK(musaMemcpy(d_k, h_k.data(), size_bytes_k, musaMemcpyHostToDevice));
  RUNTIME_CHECK(musaMemcpy(d_v, h_v.data(), size_bytes_v, musaMemcpyHostToDevice));
  RUNTIME_CHECK(musaMemset(d_o, 0, size_bytes_o));//d_o初始化为全0

  //step3: device端计算
  int Br = 32, Bc = 32;
  int grid_dim_z = (target_seq_len + Br - 1) / Br;
  dim3 block_dim(Br, Bc);
  dim3 grid_dim(query_heads, batch_size, grid_dim_z);
  size_t smem_size = (Br * Bc) * sizeof(double) + (Br * 4) * sizeof(float) + (Br * head_dim * 2 + Bc * head_dim * 2) * sizeof(float);
  
  kernel_flashAttention<T><<<grid_dim, block_dim, smem_size>>>(batch_size, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim, is_causal, d_q, d_k, d_v, d_o);
  //注意核函数返回类型只能为void

  //step4: 拷贝数据from device to host(not needed)
  RUNTIME_CHECK(musaMemcpy(h_o.data(), d_o, size_bytes_o, musaMemcpyDeviceToHost));

  //step5: free memory
  RUNTIME_CHECK(musaFree(d_q));
  RUNTIME_CHECK(musaFree(d_k));
  RUNTIME_CHECK(musaFree(d_v));
  RUNTIME_CHECK(musaFree(d_o));

  //std::cout << "h_o[0] is: " << float(h_o[0]) << std::endl;
  return;
}



// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
