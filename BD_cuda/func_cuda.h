// **************************************************************************************//
//                                                                                       //
//                              Brownian Dynamics Simulation Algorithm                   //
//                   Copyright Zilong Li, Tamar Schlick and New York University          //
//                                          April 2020                                   //
//                                                                                       //
// **************************************************************************************//


#ifndef FUNC_CUDA
#define FUNC_CUDA


#include <vector>
#include <algorithm>
#include <functional>
#include <list>
#include <iterator>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <stdio.h>
#include <cmath>
#include <cctype>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime_api.h> 

#include "mt.h"
#include "utilities.h"
#include "constants.h"

using namespace std;

extern "C++" void cuda_application_init_D_Chol(int n3);
/*
extern "C++" void cuda_application_D_Chol(int n, int n3, double* r, double a1, double a2, double* rad);

extern "C++" void cuda_application_rd(int n, int n3, double* r, double a1, double a2, double* rad, double s2dt, double* p, double* rd);

extern "C++" void cuda_application_translate(int n, int n_D3, int n3, double* r_all, double* r, double a1, double a2, double* rad, double del, double* force_global, double* t_force_global, double* LH_force_global, int n_tail3, int n_LH3, double* rd, double* r_t, double* r_lh, double* r_n, double* r_t_n, double* r_lh_n);

extern "C++" void force_and_torque_cuda_application(int n_c, int nc3, int n, int n3, int* type, double* r, double* a, double* b, double* c, double* alpha, double* beta, double* gamma, double* length, double* a_dna, double* b_dna, double* c_dna, double* alpha_p, double* beta_p, double* gamma_p, double h, double g, double s, double* phi_o, double debyell, double debye, double q_l, double k_e, double k_ex, double k_h1, double sigma_DNA_DNA, double sigma_DNA_Core, double sigma_Core_Core, double sigma_Tail_Tail, double sigma_Tail_Linker, double sigma_Tail_Core, int Nq, int Nq3, double* core_pos, double* core_q, int n_t, int n_tail, int n_tail3, double* tail_pos, int* tail_fix, int* nc_t_flag, double* r_t, double* beta_t, double* h_t, double* g_t, double* lo_t, double* beta_o_t, double* t_q, double* t_rad, int* t_grp, int* t_fix, int n_lh_n, int n_lh_g, int n_lh_c, int n_LH, int n_LH3, double* LH_g_pos, int* LH_conn, int* nc_lh_flag, double* beta_lh, double* r_lh, double* LH_q, double* LH_vdw_hh, double* LH_vdw_hc, double* LH_vdw_hl, double* LH_vdw_ht, double* LH_kstr, double* LH_kben, double* LH_streq, double* LH_betaeq, double* LH_force, double* t_force, double* force, double* torque, double* Energy);
*/

extern "C++" void cuda_application_init_data(int n_c, int nc3, int n, int n3, int* type, double* r, double* a, double* b, double* c, double* alpha, double* beta, double* gamma, double* length, double* a_dna, double* b_dna, double* c_dna, double* alpha_p, double* beta_p, double* gamma_p, double h, double g, double s, double* phi_o, double debyell, double debye, double q_l, double k_e, double k_ex, double k_h1, double sigma_DNA_DNA, double sigma_DNA_Core, double sigma_Core_Core, double sigma_Tail_Tail, double sigma_Tail_Linker, double sigma_Tail_Core, int Nq, int Nq3, double* core_pos, double* core_q, int n_t, int n_tail, int n_tail3, double* tail_pos, int* tail_fix, int* nc_t_flag, double* r_t, double* beta_t, double* h_t, double* g_t, double* lo_t, double* beta_o_t, double* t_q, double* t_rad, int* t_grp, int* t_fix, int n_lh_n, int n_lh_g, int n_lh_c, int n_LH, int n_LH3, double* LH_g_pos, int* LH_conn, int* nc_lh_flag, double* beta_lh, double* r_lh, double* LH_q, double* LH_vdw_hh, double* LH_vdw_hc, double* LH_vdw_hl, double* LH_vdw_ht, double* LH_kstr, double* LH_kben, double* LH_streq, double* LH_betaeq, double* LH_force, double* t_force, double* force, double* torque, double* Energy, double* r_all, double* rad_all);

extern "C++" void main_cuda(int n_c, int nc3, int step, int number_of_steps, double time_step, double del, int frequency_RP, int frequency_of_sampling, double h, double g, double s, double debyell, double debye, double q_l, double k_e, double k_ex, double k_h1, double sigma_DNA_DNA, double sigma_DNA_Core, double sigma_Core_Core, double sigma_Tail_Tail, double sigma_Tail_Linker, double sigma_Tail_Core, int Nq, int Nq3, int n_t, int n_lh_n, int n_lh_g, int n_lh_c, int n, int n3, int n_tail, int n_tail3, int n_LH, int n_LH3, double a1, double a2, double s2dt, double* rr, double* p, double* Energy, double* h_r, double* h_a, double* h_b, double* h_c, double* h_r_t, double* h_r_lh, double* h_rad_all, int* nc_lh_flag);

extern "C++" void free_all();

#endif

