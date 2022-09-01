// **************************************************************************************//
//											 //
//		 		Brownian Dynamics Simulation Algorithm			 //
// 		     Copyright Zilong Li, Tamar Schlick and New York University 	 //
//					    April 2020					 //
//                                                                                       //
// **************************************************************************************//


#include "constants.h"
#include "readfile.h"
#include "func.h"
#include "func_cuda.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <mpi.h>
using namespace std;

int main(int argc, char *argv[]){


	//Initial MPI
/*
        int rank, comm_size, ierr;

        ierr = MPI_Init(&argc, &argv);
        ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        ierr = MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

        int root_rank = 0;
*/
	int rank = 0;

	//Read constants from file

	std::vector<double> input_consts;

	input_consts = Read_const("setup.txt");

	double T = input_consts[0];
	int number_of_steps = (int)input_consts[1];
	int frequency_of_sampling = (int)input_consts[2];
	int frequency_RP = (int)input_consts[3];
	double sigma_DNA_DNA = input_consts[4];
        double sigma_DNA_Core = input_consts[5];
        double sigma_Core_Core = input_consts[6];
        double sigma_Tail_Tail = input_consts[7];
        double sigma_Tail_Linker = input_consts[8];
        double sigma_Tail_Core = input_consts[9];
	int Mg_flag = (int)input_consts[10];
	double Cs = input_consts[11];	
	int restart = (int)input_consts[12];
	double unbind = input_consts[13];
        double Pbind = input_consts[14];	

	//random seed

	time_t now = time(0);

        init_genrand((unsigned long) now);


	//Other constants/parameters


	double kb = 1.380649e-5; // (nm^2*kg)/(s^2*K)
	double kbt = kb*300.0; // (nm^2*kg)/s^2

	double q_l = lo * (-5.8824) *
        	(1.901912136835033e-8 * pow((Cs * 1000), 3.0) + -8.211102025728157e-6 * pow((Cs * 1000), 2.0)
        	+ 7.554037628581672e-3 * (Cs * 1000) + 3.524292543853884e-1);
	double per;
	double debye = 0.736*sqrt((Cs/0.05)*(298/T));
	double debyell;

	if (Mg_flag==0){
		per = 50;
		debyell = debye;
	}else{
		per = 30;
		debyell = 2.5;
	}
	double k_e = 0.4151*kb*300;
	double k_ex = 0.001*kbt;	
	double k_h1 = 10.0*kbt;
	double a1 = kbt/(6*PI*eta);
	double a2 = kbt/(8*PI*eta);
	double h = 100*kbt/(lo*lo);
	double hd2 = 0.5*h;
	double g = per*kbt/lo;
	double gd2 = 0.5*g;
	double s = 72.429*kbt/lo;
	double sd2 = 0.5*s;

	double time_step = 1e-12;
	double del = time_step/kbt;
	double s2dt = sqrt(2*time_step);

	//Initial structure
	
	int n, n_c, n3, nc3;
	std::vector<double> tmp_r, tmp_a, tmp_b, tmp_c, tmp_rad, tmp_Er;
	std::vector<int> tmp_type;

	Read_initial_struc(n, n_c, n3, nc3, tmp_r, tmp_a, tmp_b, tmp_c, tmp_rad, tmp_Er, tmp_type, "data_mod");
	
	//Place holder

	double r[n3], a[n3], b[n3], c[n3], Er[n3];
	int type[n];
	double rad[n];
	double alpha[n], beta[n], gamma[n], phi_o[n];
	double a_dna[nc3], b_dna[nc3], c_dna[nc3];
	double alpha_p[n_c], beta_p[n_c], gamma_p[n_c];
	double length[n];

	double rr[n3], r_var[n3], d_theta[n3];
	double force[n3], torque[n3], force_test[n3], torque_test[n3];
	double force_global[n3], torque_global[n3];
	double E[1];
	double Energy;
	double Energy_global;

	//Place holder for intermediate configurations

	double r_n[n3], a_n[n3], b_n[n3], c_n[n3], r_n_test[n3];
        double alpha_n[n], beta_n[n], gamma_n[n];
        double a_dna_n[nc3], b_dna_n[nc3], c_dna_n[nc3];
        double alpha_p_n[n_c], beta_p_n[n_c], gamma_p_n[n_c];
        double length_n[n];
	double force_n[n3], torque_n[n3];
	double force_n_global[n3], torque_n_global[n3], force_global_tmp[n3];


	//Counting
	int i, j, k;
	int i1, i2, i3, j1, j2, j3, k1, k2, k3;

	double tmp_dummy;

	if (restart==0){
		for (i=0; i<n3; i++){
			r[i] = tmp_r[i];
			a[i] = tmp_a[i];
			b[i] = tmp_b[i];
			c[i] = tmp_c[i];
			Er[i] = tmp_Er[i];
		}
	}else{
		Read_restart_ini(n3, r, a, b, c, Er);	
	}

        for (i=0; i<n; i++){
                rad[i] = tmp_rad[i];
                type[i] = tmp_type[i];
        }

	update_phi_o(n, type, phi_o);

	//Update director frame and Euler Angles
	
	update_Euler_Angle(n_c, nc3, n, n3, type, r, a, b, c, alpha, beta, gamma, length, a_dna, b_dna, c_dna, alpha_p, beta_p, gamma_p);


        for (i=0; i<n3; i++){
                a_n[i] = a[i];
                b_n[i] = b[i];
                c_n[i] = c[i];
        }


	//Place holder for charge beads, tail beads, linker histone beads.

	//Nucleosome charge beads

	int Nq, Nq3;
	std::vector<double> core_pos, core_q;

	Read_core(Nq, Nq3, core_pos, core_q, "core_data.reg.150mM");

	double core_pos_d[Nq3], core_q_d[Nq];


	//Tail beads
	
	int n_t, n_t3;
	std::vector<double> tail_pos, tail_q, tail_rad, tail_bond_c, tail_bond_v, tail_angle_c, tail_angle_v;
	std::vector<int> tail_grp, tail_fix;
	
	Read_tail(n_t, tail_pos, tail_q, tail_grp, tail_fix, tail_rad, tail_bond_c, tail_bond_v, tail_angle_c, tail_angle_v, "tail_data.mod.200mM");



	int n_tail, n_tail3; //total number of tail beads	
	int nc_t;   //number of cores have tail attached
	int nc_t_flag[n_c]; //indicate which cores have tail attached 

	nc_t = 0;
	for (i=0;i<n_c;i++){
		nc_t_flag[i]=1;
		nc_t = nc_t+nc_t_flag[i];
	}

	n_t3 = n_t*3;
	n_tail = nc_t*n_t;
	n_tail3 = n_tail*3;

	double r_t[n_tail3], beta_t[n_tail], h_t[n_tail], g_t[n_tail], lo_t[n_tail], beta_o_t[n_tail], t_q[n_tail], t_rad[n_tail], t_force[n_tail3], t_force_global[n_tail3], t_force_test[n_tail3];
	int t_grp[n_tail], t_fix[n_tail];
	double r_t_n[n_tail3], t_force_n[n_tail3], beta_t_n[n_tail], t_force_n_global[n_tail3], r_t_local[n_tail3],t_force_global_tmp[n_tail3], r_t_n_test[n_tail3];
	double tail_pos_d[n_t3];
	int tail_fix_d[n_t];

	build_tail(n, type, n_t, tail_pos, tail_q, tail_grp, tail_fix, tail_rad, tail_bond_c, tail_bond_v, tail_angle_c, tail_angle_v, r, a, b, c, n_tail, nc_t_flag, r_t, h_t, g_t, lo_t, beta_o_t, t_q, t_rad, t_grp, t_fix);

	if (restart==1){
		Read_restart_tail(n_tail3, r_t);
	}


	update_tail_beta(n_tail, r_t, beta_t, t_grp);


	//LH beads

	int n_lh_n, n_lh_g, n_lh_c;
	std::vector<double> LH_n_pos, LH_g_pos, LH_c_pos;
	std::vector<double> LH_q, LH_vdw_hh, LH_vdw_hc, LH_vdw_hl, LH_vdw_ht, LH_kstr, LH_kben, LH_streq, LH_betaeq, LH_radius;
	std::vector<int> LH_conn;


	Read_LH(n_lh_g, n_lh_n, n_lh_c, LH_g_pos, LH_n_pos, LH_c_pos, LH_conn, LH_q, LH_vdw_hh, LH_vdw_hc, LH_vdw_hl, LH_vdw_ht, LH_kstr, LH_kben, LH_streq, LH_betaeq, LH_radius, "LH_N0G6C22.in");

	double random;
        int n_b;
	int n_LH, n_LH3;
	int nc_lh;
	int nc_lh_flag[n_c], nc_lh_flag_old[n_c];
	int nlh = n_lh_n + n_lh_c;	

	nc_lh = 0;

	Read_LH_core(n_c, nc_lh, nc_lh_flag);

	n_LH = (n_lh_n + n_lh_c)*n_c;
	n_LH3 = n_LH*3;

	double LH_g_pos_d[n_lh_g*3], LH_q_d[n_lh_g+n_lh_c+n_lh_n], LH_vdw_hh_d[n_lh_g+n_lh_c+n_lh_n], LH_vdw_hc_d[n_lh_g+n_lh_c+n_lh_n], LH_vdw_hl_d[n_lh_g+n_lh_c+n_lh_n], LH_vdw_ht_d[n_lh_g+n_lh_c+n_lh_n], LH_kstr_d[n_lh_g+n_lh_c+n_lh_n], LH_kben_d[n_lh_g+n_lh_c+n_lh_n], LH_streq_d[n_lh_g+n_lh_c+n_lh_n], LH_betaeq_d[n_lh_g+n_lh_c+n_lh_n];
	int LH_conn_d[n_lh_g+n_lh_c+n_lh_n];

	double r_lh_tmp[n_LH3], lh_rad_tmp[n_LH], r_lh_n[n_LH3], r_lh_n_test[n_LH3];
//	double LH_force[n_LH3], LH_force_n[n_LH3], LH_force_global[n_LH3], LH_force_n_global[n_LH3], LH_force_global_tmp[n_LH3], LH_force_test[n_LH3];
//	double beta_lh[n_LH], beta_lh_n[n_LH];

	build_LH(n, type, n_LH, LH_n_pos, LH_c_pos, LH_radius, r, a, b, c, n_lh_n, n_lh_c, nc_lh_flag, r_lh_tmp, lh_rad_tmp);

	double* r_lh = new double[n_LH3];

        double* lh_rad = new double[n_LH];

        double r_lh_new[n_c*nlh*3], lh_rad_new[n_c*nlh*3];

        double* beta_lh = new double[n_LH];
        double* beta_lh_n = new double[n_LH];

        double* LH_force = new double[n_LH3];
        double* LH_force_n = new double[n_LH3];
        double* LH_force_global_tmp = new double[n_LH3];

        for (i=0; i<n_LH;i++){
                r_lh[i*3] = r_lh_tmp[i*3];
                r_lh[i*3+1] = r_lh_tmp[i*3+1];
                r_lh[i*3+2] = r_lh_tmp[i*3+2];
                lh_rad[i] = lh_rad_tmp[i];
        }

	if (restart == 1){
		Read_restart_LH(n_LH3, r_lh);
	}

	update_LH_beta(n_LH, n_lh_c, r_lh, beta_lh);


	// Prepare parameter for random rotation 
	
	for (i=0; i<n; i++){
		if (type[i]!=0){
			r_var[i*3] = 2.0*time_step*kbt/(8*PI*eta*125.0);
			r_var[i*3+1] = 2.0*time_step*kbt/(8*PI*eta*125.0);
			r_var[i*3+2] = 2.0*time_step*kbt/(8*PI*eta*125.0);
		}else{
			r_var[i*3] = 2.0*time_step*kbt/(4*PI*eta*r_h*r_h*lo);
			r_var[i*3+1] = 0.0;
			r_var[i*3+2] = 0.0;
		}
	}	


	double r_all[n3+n_tail3+n_LH3];

	double* rad_all=new double[n+n_tail+n_LH];

	for (i=0; i<n+n_tail+n_LH; i++){
		if (i<n){
			r_all[i*3] = r[i*3];
			r_all[i*3+1] = r[i*3+1];
			r_all[i*3+2] = r[i*3+2];
			rad_all[i] = rad[i];
		}else if (i < n+n_tail){
			r_all[i*3] = r_t[i*3-n3];
			r_all[i*3+1] = r_t[i*3-n3+1];
			r_all[i*3+2] = r_t[i*3-n3+2];
			rad_all[i] = t_rad[i-n];
		}else{
			r_all[i*3] = r_lh[i*3-n3-n_tail3];
                        r_all[i*3+1] = r_lh[i*3-n3-n_tail3+1];
                        r_all[i*3+2] = r_lh[i*3-n3-n_tail3+2];
                        rad_all[i] = lh_rad[i-n-n_tail];
		}
	}


	int n_D, n_D3;

	n_D = n+n_tail+n_LH;
        n_D3 = n_D*3;


	double* p = new double[n_D3];
        double* rd= new double[n_D3];

	if (rank==0){


		for (i=0;i<Nq3;i++){
			core_pos_d[i] = core_pos[i];
		}
		for (i=0;i<Nq;i++){
			core_q_d[i] = core_q[i];
		}

		for (i=0;i<n_t3;i++){
			tail_pos_d[i] = tail_pos[i];
		}
		for (i=0;i<n_t;i++){
			tail_fix_d[i] = tail_fix[i];
		}

		for (i=0;i<n_lh_g*3;i++){
			LH_g_pos_d[i] = LH_g_pos[i];
		}
		for (i=0;i<n_lh_g+n_lh_c+n_lh_n;i++){
			LH_q_d[i] = LH_q[i];
			LH_vdw_hh_d[i] = LH_vdw_hh[i];
			LH_vdw_hc_d[i] = LH_vdw_hc[i];
			LH_vdw_hl_d[i] = LH_vdw_hl[i];
			LH_vdw_ht_d[i] = LH_vdw_ht[i];
			LH_kstr_d[i] = LH_kstr[i];
			LH_kben_d[i] = LH_kben[i];
			LH_streq_d[i] = LH_streq[i];
			LH_betaeq_d[i] = LH_betaeq[i];
			LH_conn_d[i] = LH_conn[i];
		}

		for (i=0; i<n+n_tail+n_LH; i++){
                                if (i<n){
                                        rad_all[i] = rad[i];
                                }else if (i < n+n_tail){
                                        rad_all[i] = t_rad[i-n];
                                }else{
                                        rad_all[i] = lh_rad[i-n-n_tail];
                                }
                }

		cuda_application_init_D_Chol(n_D3);

		cuda_application_init_data(n_c, nc3, n, n3, type, r, a, b, c, alpha, beta, gamma, length, a_dna, b_dna, c_dna, alpha_p, beta_p, gamma_p, h, g, s, phi_o, debyell, debye, q_l, k_e, k_ex, k_h1, sigma_DNA_DNA, sigma_DNA_Core, sigma_Core_Core, sigma_Tail_Tail, sigma_Tail_Linker, sigma_Tail_Core, Nq, Nq3, core_pos_d, core_q_d, n_t, n_tail, n_tail3, tail_pos_d, tail_fix_d, nc_t_flag, r_t, beta_t, h_t, g_t, lo_t, beta_o_t, t_q, t_rad, t_grp, t_fix, n_lh_n, n_lh_g, n_lh_c, n_LH, n_LH3, LH_g_pos_d, LH_conn_d, nc_lh_flag, beta_lh, r_lh, LH_q_d, LH_vdw_hh_d, LH_vdw_hc_d, LH_vdw_hl_d, LH_vdw_ht_d, LH_kstr_d, LH_kben_d, LH_streq_d, LH_betaeq_d, LH_force, t_force, force, torque, E, r_all, rad_all);
	}

	//main loop

	for (int step = 0; step < number_of_steps; step++){

		if (step%100==0){
                        random = genrand_real1();
                        n_b = n_c*genrand_real1();

                        if (n_b >= n_c){
                                n_b = n_c-1;
                        }

                        for (i=0; i<n_c; i++){
                                nc_lh_flag_old[i] = nc_lh_flag[i];
                        }

                        if (nc_lh_flag[n_b]>0.5){
                                if (random <= unbind){
                                        nc_lh_flag[n_b]=0;
                                }
                        }else{
                                if (random <= Pbind){
                                        nc_lh_flag[n_b]=1;
                                }
                        }

                        n_LH = 0;
                        nc_lh=0;
                        for (i=0; i<n_c; i++){
                                if (nc_lh_flag[i]==1){
                                        n_LH=n_LH+nlh;
                                        nc_lh=nc_lh+1;
                                }
                        }
                        n_LH3 = n_LH*3;

                        build_LH(n, type, n_LH, LH_n_pos, LH_c_pos, LH_radius, r, a, b, c, n_lh_n, n_lh_c, nc_lh_flag, r_lh_new, lh_rad_new);

                        k=0;
                        k1=0;
                        for (i=0; i<n_c;i++){
                                if (nc_lh_flag[i]==1){
                                        if (nc_lh_flag_old[i]==1){
                                                for (j=0; j<nlh;j++){
                                                        r_lh_new[(k*nlh+j)*3] = r_lh[(k1*nlh+j)*3];
                                                        r_lh_new[(k*nlh+j)*3+1] = r_lh[(k1*nlh+j)*3+1];
                                                        r_lh_new[(k*nlh+j)*3+2] = r_lh[(k1*nlh+j)*3+2];
                                                }
                                                k1=k1+1;
                                        }
                                        k=k+1;
                                }else if (nc_lh_flag_old[i]==1){
                                        k1=k1+1;
                                }
                        }




                        delete [] r_lh;
                        delete [] lh_rad;

                        r_lh = new double[n_LH3];
                        lh_rad = new double[n_LH];

                        for (i=0; i<n_LH; i++){
                                r_lh[i*3]= r_lh_new[i*3];
                                r_lh[i*3+1]= r_lh_new[i*3+1];
                                r_lh[i*3+2]= r_lh_new[i*3+2];
                                lh_rad[i] = lh_rad_new[i];
                        }



                        n_D = n+n_tail+n_LH;
                        n_D3 = n_D*3;

			delete [] rad_all;

			rad_all = new double[n_D];



		}

		if (rank==0){

			for (i=0; i<n+n_tail+n_LH; i++){
                                if (i<n){
                                        rad_all[i] = rad[i];
                                }else if (i < n+n_tail){
                                        rad_all[i] = t_rad[i-n];
                                }else{
                                        rad_all[i] = lh_rad[i-n-n_tail];
                                }
                        }

			for (i = 0; i < n3; i++){
				rr[i] = rand_normal(0,r_var[i]);
			}

			
			for (i = 0; i < n3+n_tail3+n_LH3; i++){
				p[i] = rand_normal(0,1);
			}

			main_cuda(n_c, nc3, step, number_of_steps, time_step, del, frequency_RP, frequency_of_sampling, h, g, s, debyell, debye, q_l, k_e, k_ex, k_h1, sigma_DNA_DNA, sigma_DNA_Core, sigma_Core_Core, sigma_Tail_Tail, sigma_Tail_Linker, sigma_Tail_Core, Nq, Nq3, n_t, n_lh_n, n_lh_g, n_lh_c, n, n3, n_tail, n_tail3, n_LH, n_LH3, a1, a2, s2dt, rr, p, E, r, a, b, c, r_t, r_lh, rad_all, nc_lh_flag);

			//Store Energy
	
			ofstream energyfile;
			energyfile.open("Energy.txt", std::ios_base::app);
			energyfile << E[0] << endl;
			energyfile.close();


			if (step%frequency_of_sampling == 0){
				write_xyz_append(n, n3, n_c, n_tail, n_lh_g, nc_lh, nc_lh_flag, n_LH, Nq, Nq3, type, r, a, b, c, core_pos, r_t, LH_g_pos, r_lh, "out.xyz");
			}

			if (step == number_of_steps-1){
				write_restart(n3, n_tail3, n_LH3, r, a, b, c, Er, r_t, r_lh);
				ofstream LHinfile;
                                LHinfile.open("LH.in", ios::trunc);
                                for (j=0;j<n_c;j++){
                                        LHinfile << nc_lh_flag[j] << endl;
                                }
                                LHinfile.close();
			}


		}

	}

	free_all();

	time_t end = time(0);	
	
	cout << "Wall time: " << end-now << " seconds." << endl;

//        MPI_Finalize();

}

