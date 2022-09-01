#include "func_cuda.h"

double* d_D;
double* d_Chol;

//Device Matrices
int* d_type;
double* d_r;
double* d_a;
double* d_b;
double* d_c;
double* d_alpha;
double* d_beta;
double* d_gamma;
double* d_length;
double* d_a_dna;
double* d_b_dna;
double* d_c_dna;
double* d_alpha_p;
double* d_beta_p;
double* d_gamma_p;
double* d_phi_o;
double* d_force;
double* d_torque;
double* d_Energy;
double* d_core_pos;
double* d_core_q;
double* d_d_theta;
double* d_rd;

double* d_r_n;
double* d_a_n;
double* d_b_n;
double* d_c_n;
double* d_alpha_n;
double* d_beta_n;
double* d_gamma_n;
double* d_length_n;
double* d_a_dna_n;
double* d_b_dna_n;
double* d_c_dna_n;
double* d_alpha_p_n;
double* d_beta_p_n;
double* d_gamma_p_n;
double* d_force_n;
double* d_torque_n;

double* d_force_tmp;
double* d_torque_tmp;

double* d_tail_pos;
int* d_tail_fix;
int* d_nc_t_flag;
double* d_r_t;
double* d_beta_t;
double* d_h_t;
double* d_g_t;
double* d_lo_t;
double* d_beta_o_t;
double* d_t_q;
double* d_t_rad;
int* d_t_grp;
int* d_t_fix;
double* d_t_force;

double* d_r_t_n;
double* d_beta_t_n;

double* d_t_force_n;
double* d_t_force_tmp;

double* d_LH_g_pos;
int* d_LH_conn;
int* d_nc_lh_flag;
double* d_beta_lh;
double* d_r_lh;
double* d_LH_q;
double* d_LH_vdw_hh;
double* d_LH_vdw_hc;
double* d_LH_vdw_hl;
double* d_LH_vdw_ht;
double* d_LH_kstr;
double* d_LH_kben;
double* d_LH_streq;
double* d_LH_betaeq;
double* d_LH_force;

double* d_beta_lh_n;
double* d_r_lh_n;

double* d_LH_force_n;
double* d_LH_force_tmp;

double* d_force_m;
double* d_torque_m;
double* d_t_force_m;
double* d_LH_force_m;

double* d_temp_force_m;

double* d_rad_all;
double* d_r_all;

double* d_Energy_m;

__device__ void first_coord_cuda(int t, double* r, double* a, double* b, double* c, double* r_f){

        double b_m[3];
        double si, co;

        si = sin(theta);
        co = cos(theta);

        if (t==0){
                r_f[0] = r[0];
                r_f[1] = r[1];
                r_f[2] = r[2];
        }else{
                b_m[0] = -si*a[0]+co*b[0];
                b_m[1] = -si*a[1]+co*b[1];
                b_m[2] = -si*a[2]+co*b[2];

                r_f[0] = r[0] - ro*b_m[0]+d1*c[0];
                r_f[1] = r[1] - ro*b_m[1]+d1*c[1];
                r_f[2] = r[2] - ro*b_m[2]+d1*c[2];
        }

}

__device__ void second_coord_cuda(int t, double* r, double* a, double* b, double* c, double* r_s){

        if (t==0){
                r_s[0] = r[0];
                r_s[1] = r[1];
                r_s[2] = r[2];
        }else{
                r_s[0] = r[0] - (ro*b[0]+d1*c[0]);
                r_s[1] = r[1] - (ro*b[1]+d1*c[1]);
                r_s[2] = r[2] - (ro*b[2]+d1*c[2]);
        }

}

__device__ void norm_cuda(double* r, double& result){
    int i;
    double norma = 0;
    for (i = 0; i<3; i++){
        norma = norma + r[i]*r[i];
    }
    result = sqrt(norma);
}

__device__ void cross_product_cuda(double* r1, double* r2, double* product){

    product[0] = r1[1] * r2[2] - r1[2] * r2[1];
    product[1] = -1 * (r1[0] * r2[2] - r1[2] * r2[0]);
    product[2] = r1[0] * r2[1] - r1[1] * r2[0];

}

__device__ void rotate_cuda(int n, int n3, double* a, double* b, double* c, double* a_n, double* b_n, double* c_n, double* d_theta, double dt){

        double wa, wb, wc, g1, g2, g3;
        double z, z2, wa2, wb2, wc2, czt, Omczt, szt;
        int i, i1, i2, i3;

        for (i = 0; i < n; i++){
                i1 = 3*i;
                i2 = i1+1;
                i3 = i2+1;
                wa = d_theta[i1];
                wb = d_theta[i2];
                wc = d_theta[i3];
                wa2 = wa*wa;
                wb2 = wb*wb;
                wc2 = wc*wc;
                z2 = wa2 + wb2 + wc2;
                z = sqrt(z2);
                czt = cos(z*dt);
                szt = sin(z*dt);
                Omczt = 1.0 - czt;

                if (z2 > 0.0){
                        //rotation of a
                        g1 = ((wb2+wc2)*czt+wa2)/z2;
                        g2 = wa*wb*Omczt/z2 + wc*szt/z;
                        g3 = wa*wc*Omczt/z2 - wb*szt/z;
                        a_n[i1] = g1*a[i1] + g2*b[i1] + g3*c[i1];
                        a_n[i2] = g1*a[i2] + g2*b[i2] + g3*c[i2];
                        a_n[i3] = g1*a[i3] + g2*b[i3] + g3*c[i3];

                        //rotation of b
                        g1 = wa*wb*Omczt/z2 - wc*szt/z;
                        g2 = ((wa2+wc2)*czt+wb2)/z2;
                        g3 = wb*wc*Omczt/z2 + wa*szt/z;
                        b_n[i1] = g1*a[i1] + g2*b[i1] + g3*c[i1];
                        b_n[i2] = g1*a[i2] + g2*b[i2] + g3*c[i2];
                        b_n[i3] = g1*a[i3] + g2*b[i3] + g3*c[i3];

                        //rotation of c
                        g1 = wa*wc*Omczt/z2 + wb*szt/z;
                        g2 = wb*wc*Omczt/z2 - wa*szt/z;
                        g3 = ((wa2+wb2)*czt+wc2)/z2;
                        c_n[i1] = g1*a[i1] + g2*b[i1] + g3*c[i1];
                        c_n[i2] = g1*a[i2] + g2*b[i2] + g3*c[i2];
                        c_n[i3] = g1*a[i3] + g2*b[i3] + g3*c[i3];

                }else{
                        a_n[i1] = a[i1];
                        a_n[i2] = a[i2];
                        a_n[i3] = a[i3];
                        b_n[i1] = b[i1];
                        b_n[i2] = b[i2];
                        b_n[i3] = b[i3];
                        c_n[i1] = c[i1];
                        c_n[i2] = c[i2];
                        c_n[i3] = c[i3];
                }

        }

}

__device__ void Force_Stretching_cuda(double h, double* coord1, double* coord2, double lo, double* force_projection1, double* force_projection2, double& Energy){

        double force = 0.0;

        double distance = 0.0;

        distance = (coord1[0]-coord2[0])*(coord1[0]-coord2[0]) + (coord1[1]-coord2[1])*(coord1[1]-coord2[1]) + (coord1[2]-coord2[2])*(coord1[2]-coord2[2]);

	distance = sqrt(distance);

        Energy = Energy + h*(distance-lo)*(distance-lo)/2;

        force = -h*(distance-lo);

        for (int i = 0; i < 3; i++){
                force_projection1[i] = force*(coord1[i] - coord2[i])/distance;
                force_projection2[i] = -force*(coord1[i] - coord2[i])/distance;
        }

}

__device__ void Force_Bending_cuda(double g, double beta, double beta_o, double* coord1, double* coord2, double* coord3, double* force_projection1, double* force_projection2, double* force_projection3, double& Energy){

        double force = 0.0;

        double distance1, distance2;
        double norm_ri, norm_rk;

        double ji[3], jk[3], kj[3], ri[3], rk[3], product[3];

        Energy = Energy + g*(beta-beta_o)*(beta-beta_o)/2;

        force = -g*(beta-beta_o);

        distance1 = (coord1[0]-coord2[0])*(coord1[0]-coord2[0]) + (coord1[1]-coord2[1])*(coord1[1]-coord2[1]) + (coord1[2]-coord2[2])*(coord1[2]-coord2[2]);
        distance2 = (coord2[0]-coord3[0])*(coord2[0]-coord3[0]) + (coord2[1]-coord3[1])*(coord2[1]-coord3[1]) + (coord2[2]-coord3[2])*(coord2[2]-coord3[2]);

	distance1 = sqrt(distance1);
	distance2 = sqrt(distance2);

	ji[0] = coord2[0]-coord1[0];
	ji[1] = coord2[1]-coord1[1];
	ji[2] = coord2[2]-coord1[2];
	jk[0] = coord2[0]-coord3[0];
	jk[1] = coord2[1]-coord3[1];
	jk[2] = coord2[2]-coord3[2];
	kj[0] = coord3[0]-coord2[0];
	kj[1] = coord3[1]-coord2[1];
	kj[2] = coord3[2]-coord2[2];
        
	cross_product_cuda(ji, jk, product);
	for (int i = 0; i < 3; i++){
		ri[i] = product[i];
	}
        cross_product_cuda(ji, ri, product);
	for (int i = 0; i < 3; i++){
                ri[i] = product[i];
        }

        norm_cuda(ri, norm_ri);

        for (int i = 0; i < 3; i++){
                force_projection1[i] = (force/distance1)*ri[i]/norm_ri;
                force_projection2[i] = -(force/distance1)*ri[i]/norm_ri;
        }

        cross_product_cuda(ji, jk, product);
	for (int i = 0; i < 3; i++){
                rk[i] = product[i];
        }
        cross_product_cuda(kj, rk, product);
	for (int i = 0; i < 3; i++){
                rk[i] = product[i];
        }

        norm_cuda(rk, norm_rk);

        for (int j = 0; j < 3; j++){
                force_projection2[j] = force_projection2[j] - (force/distance2)*rk[j]/norm_rk;
                force_projection3[j] = (force/distance2)*rk[j]/norm_rk;
        }

}


__device__ void Bending_force_projection_cuda(double g, double beta, double beta_b, double length, double* a_f, double* a_b, double* a, double* force_projection1, double* force_projection2, double& Energy){

        double Ai[3], Bi[3];
        double c1, c2, g1, g2;

        Energy = Energy + g*(beta-beta_b)*(beta-beta_b)/2;

        if (beta >= 1e-10){
                g1 = beta/(sin(beta)*length);
        }else{
                g1 = 1.0/length;
        }
        c1 = cos(beta);

        Ai[0] = g1*(a_f[0]-c1*a[0]);
        Ai[1] = g1*(a_f[1]-c1*a[1]);
        Ai[2] = g1*(a_f[2]-c1*a[2]);

        if (beta_b >= 1e-10){
                g2 = beta_b/(sin(beta_b)*length);
        }else{
                g2 = 1.0/length;
        }
        c2 = cos(beta_b);

        Bi[0] = g2*(a_b[0]-c2*a[0]);
        Bi[1] = g2*(a_b[1]-c2*a[1]);
        Bi[2] = g2*(a_b[2]-c2*a[2]);

        for (int i = 0; i < 3; i++){
                force_projection1[i] = -g*(Ai[i]+Bi[i]);
                force_projection2[i] = g*(Ai[i]+Bi[i]);
        }

}

__device__ void Twisting_force_projection_cuda(double s, double alpha, double beta, double gamma, double phi_o, double length, double alpha_b, double beta_b, double gamma_b, double phi_o_b, double gamma_n, double* b, double* c, double* force_projection1, double* force_projection2, double& Energy){

        double Chi[3], Zhi[3];
        double g1, g2, c1, c2, s1, s2;

        Energy = Energy + s*((alpha+gamma-phi_o)*(alpha+gamma-phi_o))/2;

        g1 = (alpha+gamma-phi_o)*tan(0.5*beta)/length;
        c1 = cos(alpha);
        s1 = sin(alpha);

        g2 = (alpha_b+gamma_b-phi_o_b)*tan(0.5*beta_b)/length;
        c2 = cos(gamma_n);
        s2 = sin(gamma_n);

        Chi[0] = g1*(c1*c[0]-s1*b[0]);
        Chi[1] = g1*(c1*c[1]-s1*b[1]);
        Chi[2] = g1*(c1*c[2]-s1*b[2]);
        Zhi[0] = g2*(c2*c[0]+s2*b[0]);
        Zhi[1] = g2*(c2*c[1]+s2*b[1]);
        Zhi[2] = g2*(c2*c[2]+s2*b[2]);

        for (int i = 0; i < 3; i++){
                force_projection1[i] = s*(Chi[i]+Zhi[i]);
                force_projection2[i] = -s*(Chi[i]+Zhi[i]);
        }
}

__device__ void Force_Electrostatics_cuda(double q1, double q2, double epslon, double kappa, double* coord1, double* coord2, double* force_projection1, double* force_projection2, double& Energy){

        double force = 0.0;
        double Rcut = 7.0;
        double distance = 0.0;
	double temp;

        distance = (coord1[0]-coord2[0])*(coord1[0]-coord2[0]) + (coord1[1]-coord2[1])*(coord1[1]-coord2[1]) + (coord1[2]-coord2[2])*(coord1[2]-coord2[2]);

	distance = sqrt(distance);

        if (distance < Rcut){
		temp = -kappa*distance;
                force = ((q1*q2*(kappa*distance+1))/(4*PI*epslon*distance*distance))*exp(temp);
                Energy = Energy + (q1*q2/(4*PI*epslon*distance))*exp(temp);
        }

        for (int i = 0; i < 3; i++){
                force_projection1[i] = force*(coord1[i] - coord2[i])/distance;
                force_projection2[i] = -force*(coord1[i] - coord2[i])/distance;
                if (force_projection1[i]>0.9){ force_projection1[i]=0.9; }
                if (force_projection2[i]>0.9){ force_projection2[i]=0.9; }
                if (force_projection1[i]<-0.9){ force_projection1[i]=-0.9; }
                if (force_projection2[i]<-0.9){ force_projection2[i]=-0.9; }
        }

}

__device__ void Force_Exclude_Volume_cuda(double k_ev, double sigma, double* coord1, double* coord2, double* force_projection1, double* force_projection2, double& Energy){

        double force = 0.0;
        double vdw_cut = 4.0;
        double distance = 0.0;

        distance = (coord1[0]-coord2[0])*(coord1[0]-coord2[0]) + (coord1[1]-coord2[1])*(coord1[1]-coord2[1]) + (coord1[2]-coord2[2])*(coord1[2]-coord2[2]);

	distance = sqrt(distance);

        if (distance < vdw_cut){
                force = -k_ev*((6*pow(sigma,6))/pow(distance,7) - (12*pow(sigma,12))/pow(distance,13));
                Energy = Energy + k_ev*(pow(sigma/distance,12) - pow(sigma/distance,6));
        }

        for (int i = 0; i < 3; i++){
                force_projection1[i] = force*(coord1[i] - coord2[i])/distance;
                force_projection2[i] = -force*(coord1[i] - coord2[i])/distance;
                if (force_projection1[i]>0.9){ force_projection1[i]=0.9; }
                if (force_projection2[i]>0.9){ force_projection2[i]=0.9; }
                if (force_projection1[i]<-0.9){ force_projection1[i]=-0.9; }
                if (force_projection2[i]<-0.9){ force_projection2[i]=-0.9; }
        }

}

__device__ void Force_Ele_Vdw_cuda(double q1, double q2, double epslon, double kappa, double k_ev, double sigma, double* coord1, double* coord2, double* force_projection1, double* force_projection2, double& Energy){

        double force = 0.0;
        double force2 = 0.0;
        double Rcut = 7.0;
        double vdw_cut = 4.0;
        double distance = 0.0;
	double temp;

        distance = (coord1[0]-coord2[0])*(coord1[0]-coord2[0]) + (coord1[1]-coord2[1])*(coord1[1]-coord2[1]) + (coord1[2]-coord2[2])*(coord1[2]-coord2[2]);

	distance = sqrt(distance);

        if (distance < Rcut){
		temp = -kappa*distance;
                force = ((q1*q2*(kappa*distance+1))/(4*PI*epslon*distance*distance))*exp(temp);
                Energy = Energy + (q1*q2/(4*PI*epslon*distance))*exp(temp);
        }

        if (distance < vdw_cut){
                force2 = -k_ev*((6*pow(sigma,6))/pow(distance,7) - (12*pow(sigma,12))/pow(distance,13));
                Energy = Energy + k_ev*(pow(sigma/distance,12) - pow(sigma/distance,6));
        }

        for (int i = 0; i < 3; i++){
                force_projection1[i] = (force+force2)*(coord1[i] - coord2[i])/distance;
                force_projection2[i] = -(force+force2)*(coord1[i] - coord2[i])/distance;
                if (force_projection1[i]>0.9){ force_projection1[i]=0.9; }
                if (force_projection2[i]>0.9){ force_projection2[i]=0.9; }
                if (force_projection1[i]<-0.9){ force_projection1[i]=-0.9; }
                if (force_projection2[i]<-0.9){ force_projection2[i]=-0.9; }
        }

}

__device__ void torque_due_to_force_cuda(double* force, double* coord_f, double* coord_c, double* a, double* b, double* c, double* torque){

        //Given exact coordinates of the bead with force applied (coord_f) and the exact coordinate of the center bead (coord_c)

        double fa, fb, fc;
        double comp[3];

        comp[0] = (coord_f[0]-coord_c[0])*a[0] + (coord_f[1]-coord_c[1])*a[1] + (coord_f[2]-coord_c[2])*a[2];
        comp[1] = (coord_f[0]-coord_c[0])*b[0] + (coord_f[1]-coord_c[1])*b[1] + (coord_f[2]-coord_c[2])*b[2];
        comp[2] = (coord_f[0]-coord_c[0])*c[0] + (coord_f[1]-coord_c[1])*c[1] + (coord_f[2]-coord_c[2])*c[2];


        fa = a[0]*force[0] + a[1]*force[1] + a[2]*force[2];
        fb = b[0]*force[0] + b[1]*force[1] + b[2]*force[2];
        fc = c[0]*force[0] + c[1]*force[1] + c[2]*force[2];

        torque[0] = fc*comp[1] - fb*comp[2];
        torque[1] = fa*comp[2] - fc*comp[0];
        torque[2] = fb*comp[0] - fa*comp[1];

}

__device__ void torque_due_to_force_relative_cuda(double* force, double* comp, double* a, double* b, double* c, double* torque){

        //Given relative position (comp) of the bead with force applied

        double fa, fb, fc;

        fa = a[0]*force[0] + a[1]*force[1] + a[2]*force[2];
        fb = b[0]*force[0] + b[1]*force[1] + b[2]*force[2];
        fc = c[0]*force[0] + c[1]*force[1] + c[2]*force[2];

        torque[0] = fc*comp[1] - fb*comp[2];
        torque[1] = fa*comp[2] - fc*comp[0];
        torque[2] = fb*comp[0] - fa*comp[1];

}

__global__ void Diffusion_Tensor_CUDA(int n, int n3, double* r, double a1, double a2, double* rad, double* D, double* Chol){

	// Calculate Row and Column
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;

	// indexing

	double sij[3];
        double ssq, s, f, f1, f2;
        int i, j, k, l;

	// Diffusion Tensor

	i = row / 3;
	j = column / 3;
	k = row % 3;
	l = column % 3;

	if (row < n3 and column < n3){

		if (row == column){
			D[row*n3+column] = a1/rad[i];
			Chol[row*n3+column] = a1/rad[i];
		} else if (i==j and k!=l){
			D[row*n3+column] = 0;
			Chol[row*n3+column] = 0;
		} else {
			sij[0] = r[3*i] - r[3*j];
	                sij[1] = r[3*i+1] - r[3*j+1];
        	        sij[2] = r[3*i+2] - r[3*j+2];
                	ssq = sij[0]*sij[0] + sij[1]*sij[1] + sij[2]*sij[2];
	                s = sqrt(ssq);
			
			if (s >= rad[i]+rad[j]){
				f = (rad[i]*rad[i]+rad[j]*rad[j])/ssq;
				f1 = 1.0 + f/3;
                        	f2 = 1.0 -f;
				if (k==l){
					D[row*n3+column] = (a2/s)*(f1+f2*sij[k]*sij[k]/ssq);
					Chol[row*n3+column] = (a2/s)*(f1+f2*sij[k]*sij[k]/ssq);
				} else {
					D[row*n3+column] = (a2/s)*(f2*sij[k]*sij[l]/ssq);
					Chol[row*n3+column] = (a2/s)*(f2*sij[k]*sij[l]/ssq);
				}

			}else{
				s = pow((rad[i]*rad[i]*rad[i]+rad[j]*rad[j]*rad[j])/2.0, 1.0/3);
				if (k==l){
					D[row*n3+column] = (a1/s)*(1.0-9.0*sqrt(ssq)/(32*s) + 3.0*sij[k]*sij[k]/(32.0*s*sqrt(ssq)));
					Chol[row*n3+column] = (a1/s)*(1.0-9.0*sqrt(ssq)/(32*s) + 3.0*sij[k]*sij[k]/(32.0*s*sqrt(ssq)));
				} else {
					D[row*n3+column] = (a1/s)*(3.0*sij[k]*sij[l]/(32.0*s*sqrt(ssq)));
					Chol[row*n3+column] = (a1/s)*(3.0*sij[k]*sij[l]/(32.0*s*sqrt(ssq)));
				}
			}
		}

	}
}

__global__ void Cholesky_Decomposition_CUDA(double* D, int n3, double* Chol){

	// Calculate Row and Column
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int column = blockIdx.x * blockDim.x + threadIdx.x;

	// indexing

	double ssq, f;
	int i,j,k;

	//Cholesky decomposition

	if (row < n3 and column < n3){

		Chol[row*n3+column] = 0;
		
                if (row == 0 and column == 0){
                        Chol[0] = sqrt(D[0]);
                
                	for (i=1; i<n3; i++){
                                
                                Chol[i*n3] = D[i*n3]/Chol[0];
                        
                                for (j=1;j<i;j++){
                                        ssq = 0.0;
                                        for (k=0;k<j;k++){
                                                ssq += Chol[i*n3+k]*Chol[j*n3+k];
                                        }
                                      
                                        Chol[i*n3+j] = (D[i*n3+j]-ssq)/Chol[j*n3+j];
                                }

                                ssq = 0.0;

                                for (k=0;k<i;k++){
                                        ssq += Chol[i*n3+k]*Chol[i*n3+k];
                                }
                                
                                f = D[i*n3+i] - ssq;

                                if (f < 0){
                                        f = -f;
                                }
                                Chol[i*n3+i] = sqrt(f);
                        }
                }

	}


}

__global__ void Cholesky_Decomposition_mod(int n3, double* Chol){

        // Calculate Row and Column
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int column = blockIdx.x * blockDim.x + threadIdx.x;


        if (row < n3 and column < n3){
                if (row < column){
                        Chol[row*n3+column] = 0.0;
                }
		if (isnan(Chol[row*n3+column])){
			Chol[row*n3+column] = 0.0;
		}
        }

}



__global__ void rd_cal(int n3, double* rd, double* Chol, double* p, double s2dt){

	int column = blockIdx.x * blockDim.x + threadIdx.x;

	if (column < n3){

		rd[column] = 0.0;
	
		for (int k = 0; k <= column; k++){
			rd[column] += s2dt*Chol[column*n3+k]*p[k];
		}

	}
}


__global__ void translation_cal(int n_D3, int n3, int n_tail3, int n_LH3, double* r, double* r_n, double* rd, double* r_t, double* r_t_n, double* r_lh, double* r_lh_n, double del, double* force_global, double* t_force_global, double* LH_force_global, double* D){

	int column = blockIdx.x * blockDim.x + threadIdx.x;

	int k;

	if (column < n_D3){

	
                if (column < n3){
                        r_n[column] = r[column] + rd[column];
                        for (k = 0; k < n3; k++){
                                r_n[column] = r_n[column] + del*D[column*n_D3+k]*force_global[k];
			}
                }else if (column < n3+n_tail3){
                        r_t_n[column-n3] = r_t[column-n3] + rd[column];
                        for (k = n3; k < n3+n_tail3; k++){
                                r_t_n[column-n3] = r_t_n[column-n3] + del*D[column*n_D3+k]*t_force_global[k-n3];
                        }
                }else{
                        r_lh_n[column-n3-n_tail3] = r_lh[column-n3-n_tail3] + rd[column];
                        for (k = n3+n_tail3; k < n3+n_tail3+n_LH3; k++){
                                r_lh_n[column-n3-n_tail3] = r_lh_n[column-n3-n_tail3] + del*D[column*n_D3+k]*LH_force_global[k-n3-n_tail3];
                        }
                }
        

	}

}

__global__ void rotation_cal(int n, double* d_theta, int* type, double time_step, double* torque, double* rr, double* a, double* b, double* c, double* a_n, double* b_n, double* c_n){

	int j = blockIdx.x * blockDim.x + threadIdx.x;

	int j1, j2, j3;

	if (j < n){
		j1 = 3*j;
                j2 = j1+1;
                j3 = j2+1;
                if (type[j] != 0){
                	d_theta[j1] = time_step*torque[j1]/(8*PI*eta*125.0) + rr[j1];
                        d_theta[j2] = time_step*torque[j2]/(8*PI*eta*125.0) + rr[j2];
                        d_theta[j3] = time_step*torque[j3]/(8*PI*eta*125.0) + rr[j3];
                }else{
                        d_theta[j1] = time_step*torque[j1]/(4*PI*eta*r_h*r_h*lo) + rr[j1];
                        d_theta[j2] = 0.0;
                        d_theta[j3] = 0.0;
                }	
	}

	rotate_cuda(n, n*3, a, b, c, a_n, b_n, c_n, d_theta, 1.0);

}

__global__ void update_Euler_Angle_cuda(int n_c, int nc3, int n, int n3, int* type, double* r, double* a, double* b, double* c, double* alpha, double* beta, double* gamma, double* length, double* a_dna, double* b_dna, double* c_dna, double* alpha_p, double* beta_p, double* gamma_p){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
        double r_forw[3];
        double mi;
        double da[3], a_old[3];
        double a_m[3], b_m[3];
        double Ac, apg, f1, f2, ada, bda, si, co;
        double sa, ca, sb, cb, sg, cg;
        double R21, R22, R23, R31, R32, R33;
        int i1,i2,i3, if1,if2,if3, ic,ic1,ic2,ic3;
        int count;

        si = sin(theta);
        co = cos(theta);

        if (i < n-1){
                i1 = 3*i;
                i2 = i1+1;
                i3 = i2+1;
                if1 = i1+3;
                if2 = if1+1;
                if3 = if2+1;

                if (type[i] == 0){
                        a_old[0] = a[i1];
                        a_old[1] = a[i2];
                        a_old[2] = a[i3];
                        if (type[i+1] == 0){
                                r_forw[0] = r[if1] - r[i1];
                                r_forw[1] = r[if2] - r[i2];
                                r_forw[2] = r[if3] - r[i3];
                        }else{
                                b_m[0] = -si*a[if1] + co*b[if1];
                                b_m[1] = -si*a[if2] + co*b[if2];
                                b_m[2] = -si*a[if3] + co*b[if3];
                                r_forw[0] = r[if1] - ro*b_m[0] + d1*c[if1] - r[i1];
                                r_forw[1] = r[if2] - ro*b_m[1] + d1*c[if2] - r[i2];
                                r_forw[2] = r[if3] - ro*b_m[2] + d1*c[if3] - r[i3];
                        }
                        length[i] = r_forw[0]*r_forw[0] + r_forw[1]*r_forw[1] + r_forw[2]*r_forw[2];
			length[i] = sqrt(length[i]);
                        mi = 1.0/length[i];
                        a[i1] = mi*r_forw[0];
                        a[i2] = mi*r_forw[1];
                        a[i3] = mi*r_forw[2];
                        da[0] = a[i1] - a_old[0];
                        da[1] = a[i2] - a_old[1];
                        da[2] = a[i3] - a_old[2];

                        bda = b[i1]*da[0] +b[i2]*da[1] + b[i3]*da[2];
                        b[i1] = b[i1] - bda*a_old[0];
                        b[i2] = b[i2] - bda*a_old[1];
                        b[i3] = b[i3] - bda*a_old[2];

                        bda = b[i1]*a[i1] + b[i2]*a[i2] + b[i3]*a[i3];
                        b[i1] = b[i1] - bda*a[i1];
                        b[i2] = b[i2] - bda*a[i2];
                        b[i3] = b[i3] - bda*a[i3];

                        mi = 1.0/sqrt(b[i1]*b[i1] + b[i2]*b[i2] + b[i3]*b[i3]);
                        b[i1] = mi*b[i1];
                        b[i2] = mi*b[i2];
                        b[i3] = mi*b[i3];

                        c[i1] = a[i2]*b[i3] - a[i3]*b[i2];
                        c[i2] = a[i3]*b[i1] - a[i1]*b[i3];
                        c[i3] = a[i1]*b[i2] - a[i2]*b[i1];
                }else{
                        r_forw[0] = r[if1] - (r[i1]-ro*b[i1]-d1*c[i1]);
                        r_forw[1] = r[if2] - (r[i2]-ro*b[i2]-d1*c[i2]);
                        r_forw[2] = r[if3] - (r[i3]-ro*b[i3]-d1*c[i3]);
                        length[i] = r_forw[0]*r_forw[0] + r_forw[1]*r_forw[1] + r_forw[2]*r_forw[2];
			length[i] = sqrt(length[i]);
                }
        }

	if (i < n-1){
                i1 = 3*i;
                i2 = i1+1;
                i3 = i2+1;
                if1 = i1+3;
                if2 = if1+1;
                if3 = if2+1;

                if (type[i]==0){
                        if(type[i+1]==0){
                                ada = a[i1]*a[if1]+a[i2]*a[if2]+a[i3]*a[if3];
                                if (ada > 1.0) ada = 1.0;
                                if (ada < -1.0) ada = -1.0;
                                beta[i] = acos(ada);
                                sb = sin(beta[i]);
                                if (beta[i] > 1e-10){
                                        f1 = (a[if1]*b[i1]+a[if2]*b[i2]+a[if3]*b[i3])/sb;
                                }else{
                                        f1 = (b[if1]*b[i1]+b[if2]*b[i2]+b[if3]*b[i3]);
                                }
                                if (f1 > 1.0) f1 = 1.0;
                                if (f1 < -1.0) f1 = -1.0;
                                Ac = acos(f1);
                                f2 = a[if1]*c[i1] + a[if2]*c[i2] + a[if3]*c[i3];
                                if (f2 >= 0){
                                        alpha[i] = Ac;
                                }else{
                                        alpha[i] = -Ac;
                                }

                                f1 = (b[i1]*b[if1]+b[i2]*b[if2]+b[i3]*b[if3]+c[i1]*c[if1]+c[i2]*c[if2]+c[i3]*c[if3])/(1.0 + ada);
                                if (f1 > 1.0) f1 = 1.0;
                                if (f1 < -1.0) f1 = -1.0;
                                apg = acos(f1);
                                f2 = (c[i1]*b[if1]+c[i2]*b[if2]+c[i3]*b[if3]-(b[i1]*c[if1]+b[i2]*c[if2]+b[i3]*c[if3]))/(1.0 + ada);
                                if (f2 >= 0.0){
                                        gamma[i] = apg - alpha[i];
                                }else{
                                        gamma[i] = -apg - alpha[i];
                                }
                        }else{
                                a_m[0] = co*a[if1] + si*b[if1];
                                a_m[1] = co*a[if2] + si*b[if2];
                                a_m[2] = co*a[if3] + si*b[if3];
                                b_m[0] = -si*a[if1] + co*b[if1];
                                b_m[1] = -si*a[if2] + co*b[if2];
                                b_m[2] = -si*a[if3] + co*b[if3];

                                ada = a[i1]*a_m[0]+a[i2]*a_m[1]+a[i3]*a_m[2];
                                if (ada > 1.0) ada = 1.0;
                                if (ada < -1.0) ada = -1.0;
                                beta[i] = acos(ada);
                                sb = sin(beta[i]);
                                if (beta[i] > 1e-10){
                                        f1 = (a_m[0]*b[i1]+a_m[1]*b[i2]+a_m[2]*b[i3])/sb;
                                }else{
                                        f1 = (b_m[0]*b[i1]+b_m[1]*b[i2]+b_m[2]*b[i3]);
                                }
                                if (f1 > 1.0) f1 = 1.0;
                                if (f1 < -1.0) f1 = -1.0;
                                Ac = acos(f1);
                                f2 = a_m[0]*c[i1] + a_m[1]*c[i2] + a_m[2]*c[i3];
                                if (f2 >= 0){
                                        alpha[i] = Ac;
                                }else{
                                        alpha[i] = -Ac;
                                }

                                f1 = (b[i1]*b_m[0]+b[i2]*b_m[1]+b[i3]*b_m[2]+c[i1]*c[if1]+c[i2]*c[if2]+c[i3]*c[if3])/(1.0 + ada);
                                if (f1 > 1.0) f1 = 1.0;
                                if (f1 < -1.0) f1 = -1.0;
                                apg = acos(f1);
                                f2 = (c[i1]*b_m[0]+c[i2]*b_m[1]+c[i3]*b_m[2]-(b[i1]*c[if1]+b[i2]*c[if2]+b[i3]*c[if3]))/(1.0 + ada);
                                if (f2 >= 0.0){
                                        gamma[i] = apg - alpha[i];
                                }else{
                                        gamma[i] = -apg - alpha[i];
                                }
                        }
                }else{
                        ic = 0;
                        for (count = 0; count <= i; count++){
                                if(type[count]==1) ic=ic+1;
                        }
                        ic = ic-1;
                        ic1 = 3*ic;
                        ic2 = ic1+1;
                        ic3 = ic2+1;

                        a_dna[ic1] = (r[if1] - (r[i1]-ro*b[i1]-d1*c[i1]))/length[i];
                        a_dna[ic2] = (r[if2] - (r[i2]-ro*b[i2]-d1*c[i2]))/length[i];
                        a_dna[ic3] = (r[if3] - (r[i3]-ro*b[i3]-d1*c[i3]))/length[i];

                        cb = a[i1]*a_dna[ic1] + a[i2]*a_dna[ic2] + a[i3]*a_dna[ic3];
                        if (cb > 1.0) cb = 1.0;
                        if (cb < -1.0) cb = -1.0;
                        beta_p[ic] = acos(cb);
                        sb = sin(beta_p[ic]);
                        if (beta_p[ic] >= 1e-10){
                                b_m[0] = (a_dna[ic1]-cb*a[i1])/sb;
                                b_m[1] = (a_dna[ic2]-cb*a[i2])/sb;
                                b_m[2] = (a_dna[ic3]-cb*a[i3])/sb;
                                ca = b_m[0]*b[i1]+b_m[1]*b[i2]+b_m[2]*b[i3];
                                if (ca > 1.0) ca = 1.0;
                                if (ca < -1.0) ca = -1.0;
                                Ac = acos(ca);
                                f1 = a_dna[ic1]*c[i1]+a_dna[ic2]*c[i2]+a_dna[ic3]*c[i3];
                                if (f1 >= 0){
                                        alpha_p[ic] = Ac;
                                }else{
                                        alpha_p[ic] = -Ac;
                                }
                                gamma_p[ic] = -alpha_p[ic];
                                sa = sin(alpha_p[ic]);
                                sg = sin(gamma_p[ic]);
                                cg = cos(gamma_p[ic]);
                                R21 = -cg*sb;
                                R22 = cg*cb*ca-sg*sa;
                                R23 = cg*cb*sa+sg*ca;

                                b_dna[ic1] = R21*a[i1] + R22*b[i1] + R23*c[i1];
                                b_dna[ic2] = R21*a[i2] + R22*b[i2] + R23*c[i2];
                                b_dna[ic3] = R21*a[i3] + R22*b[i3] + R23*c[i3];

                                R31 = sg*sb;
                                R32 = -sg*cb*ca-cg*sa;
                                R33 = -sg*cb*sa+cg*ca;

                                c_dna[ic1] = R31*a[i1]+R32*b[i1]+R33*c[i1];
                                c_dna[ic2] = R31*a[i2]+R32*b[i2]+R33*c[i2];
                                c_dna[ic3] = R31*a[i3]+R32*b[i3]+R33*c[i3];


                        }else{
                                b_dna[ic1] = b[i1];
                                b_dna[ic2] = b[i2];
                                b_dna[ic3] = b[i3];
                                c_dna[ic1] = c[i1];
                                c_dna[ic2] = c[i2];
                                c_dna[ic3] = c[i3];
                        }
                        ada = a_dna[ic1]*a[if1]+a_dna[ic2]*a[if2]+a_dna[ic3]*a[if3];
                        if (ada > 1.0) ada = 1.0;
                        if (ada < -1.0) ada = -1.0;
                        beta[i] = acos(ada);
                        sb  = sin(beta[i]);
                        if (beta[i] >= 1e-10){
                                f1 = (a[if1]*b_dna[ic1]+a[if2]*b_dna[ic2]+a[if3]*b_dna[ic3])/sb;
                        }else{
                                f1 = (b[if1]*b_dna[ic1]+b[if2]*b_dna[ic2]+b[if3]*b_dna[ic3]);
                        }
                        if (f1 > 1.0) f1 =1.0;
                        if (f1 < -1.0) f1 = -1.0;
                        Ac = acos(f1);
                        f2 = a[if1]*c_dna[ic1]+a[if2]*c_dna[ic2]+a[if3]*c_dna[ic3];
                        if (f2 >= 0){
                                alpha[i] = Ac;
                        }else{
                                alpha[i] = -Ac;
                        }

                        f1 = (b_dna[ic1]*b[if1]+b_dna[ic2]*b[if2]+b_dna[ic3]*b[if3]+c_dna[ic1]*c[if1]+c_dna[ic2]*c[if2]+c_dna[ic3]*c[if3])/(1.0+ada);
                        if (f1 > 1.0) f1 = 1.0;
                        if (f1 < -1.0) f1 = -1.0;
                        apg = acos(f1);
                        f2 = (c_dna[ic1]*b[if1]+c_dna[ic2]*b[if2]+c_dna[ic3]*b[if3]-(b_dna[ic1]*c[if1]+b_dna[ic2]*c[if2]+b_dna[ic3]*c[if3]))/(1.0+ada);
                        if (f2 >= 0.0){
                                gamma[i] = apg - alpha[i];
                        }else{
                                gamma[i] = -apg - alpha[i];
                        }

                }

        }



}



__global__ void mechanical_force_and_torque_cuda(int maxrtlh, int n_c, int nc3, int n, int n3, int* type, double* r, double* a, double* b, double* c, double* alpha, double* beta, double* gamma, double* length, double* a_dna, double* b_dna, double* c_dna, double* alpha_p, double* beta_p, double* gamma_p, double h, double g, double s, double* phi_o, double* force, double* torque, double* Energy, double* force_m, double* torque_m, double* Energy_m){

	int ic, count;
        int i1, i2, i3, ib1, ib2, ib3, ic1, ic2, ic3, im1;
        double c1, s1, si, co;
        double a_m[3];
        double alpha_b, beta_b, gamma_b, phi_o_b, gamma_n;
        double a_f[3], a_b[3], a_o[3], b_o[3], c_o[3];
        double mag;
        double ada, adb, adc, cda, bda, cdb, bdb, cdc, bdc;

        double torque1[3], torque2[3]; 
        double force_projection1[3], force_projection2[3]; 
        double r_f[3], r_s[3]; 

        double r_tmp1[3], r_tmp2[3], a_tmp1[3], a_tmp2[3], b_tmp1[3], b_tmp2[3], c_tmp1[3], c_tmp2[3];


        si = sin(theta);
        co = cos(theta);

	int column = blockIdx.x * blockDim.x + threadIdx.x;

	//Stretching

        if (column<n-1){

                for (int xi = 0; xi <3; xi++){
                        r_tmp1[xi] = r[column*3+3+xi];
                        r_tmp2[xi] = r[column*3+xi];
                        a_tmp1[xi] = a[column*3+3+xi];
                        a_tmp2[xi] = a[column*3+xi];
                        b_tmp1[xi] = b[column*3+3+xi];
                        b_tmp2[xi] = b[column*3+xi];
                        c_tmp1[xi] = c[column*3+3+xi];
                        c_tmp2[xi] = c[column*3+xi];
                }

                first_coord_cuda (type[column+1], r_tmp1, a_tmp1, b_tmp1, c_tmp1, r_f);
		second_coord_cuda (type[column], r_tmp2, a_tmp2, b_tmp2, c_tmp2, r_s);

                Force_Stretching_cuda (h,r_s,r_f,lo,force_projection1,force_projection2, Energy_m[column]);

                force_m[column*2*maxrtlh*3] += force_projection1[0];
                force_m[column*2*maxrtlh*3+1] += force_projection1[1];
                force_m[column*2*maxrtlh*3+2] += force_projection1[2];

		force_m[(column+1)*2*maxrtlh*3+3] += force_projection2[0];
                force_m[(column+1)*2*maxrtlh*3+4] += force_projection2[1];
                force_m[(column+1)*2*maxrtlh*3+5] += force_projection2[2];

                torque_due_to_force_cuda (force_projection1, r_s, r_tmp2, a_tmp2, b_tmp2, c_tmp2, torque1);
                torque_due_to_force_cuda (force_projection2, r_f, r_tmp1, a_tmp1, b_tmp1, c_tmp1, torque2);

                torque_m[column*2*maxrtlh*3] += torque1[0];
                torque_m[column*2*maxrtlh*3+1] += torque1[1];
                torque_m[column*2*maxrtlh*3+2] += torque1[2];

		torque_m[(column+1)*2*maxrtlh*3+3] += torque2[0];
                torque_m[(column+1)*2*maxrtlh*3+4] += torque2[1];
                torque_m[(column+1)*2*maxrtlh*3+5] += torque2[2];

        }

	__threadfence();

	//Bending
        if (column < n-1){

                ic = 0;
                for (count = 0; count <= column; count++){
                        if(type[count]==1) ic=ic+1;
                }
                ic = ic-1;
                ic1 = 3*ic;
                ic2 = ic1+1;
                ic3 = ic2+1;

                for (int xi = 0; xi <3; xi++){
                        r_tmp1[xi] = r[column*3+3+xi];
                        r_tmp2[xi] = r[column*3+xi];
                        a_tmp1[xi] = a[column*3+3+xi];
                        a_tmp2[xi] = a[column*3+xi];
                        b_tmp1[xi] = b[column*3+3+xi];
                        b_tmp2[xi] = b[column*3+xi];
                        c_tmp1[xi] = c[column*3+3+xi];
                        c_tmp2[xi] = c[column*3+xi];
                }

                first_coord_cuda(type[column+1], r_tmp1, a_tmp1, b_tmp1, c_tmp1, r_f);
                second_coord_cuda(type[column], r_tmp2, a_tmp2, b_tmp2, c_tmp2, r_s);

                if (type[column]==0){
                        if (type[column+1]==0){
                                for (int xi = 0; xi<3; xi++){
                                        a_f[xi] = a[column*3+3+xi];
                                }
                        }else{
                                a_m[0] = co*a[column*3+3] + si*b[column*3+3];
                                a_m[1] = co*a[column*3+4] + si*b[column*3+4];
                                a_m[2] = co*a[column*3+5] + si*b[column*3+5];
                                for (int xi = 0; xi<3; xi++){
                                        a_f[xi] = a_m[xi];
                                }
                        }
                        if (type[column-1]==0){
                                for (int xi = 0; xi<3; xi++){
                                        a_b[xi] = a[column*3-3+xi];
                                }
                        }else{
                                for (int xi = 0; xi<3; xi++){
                                        a_b[xi] = a_dna[ic1+xi];
                                }
                        }
                        for (int xi = 0; xi<3; xi++){
                                a_o[xi] = a[column*3+xi];
                        }
                        beta_b = beta[column-1];

                }else{
                        for (int xi = 0; xi<3; xi++){
                                a_f[xi] = a[column*3+3+xi];
                                a_b[xi] = a[column*3+xi];
                                a_o[xi] = a_dna[ic1+xi];
                        }
                        beta_b = beta_p[ic];
                }

                Bending_force_projection_cuda(g, beta[column], beta_b, length[column], a_f, a_b, a_o, force_projection1, force_projection2, Energy_m[column]);

		force_m[column*2*maxrtlh*3] += force_projection1[0];
                force_m[column*2*maxrtlh*3+1] += force_projection1[1];
                force_m[column*2*maxrtlh*3+2] += force_projection1[2];

		force_m[(column+1)*2*maxrtlh*3+3] += force_projection2[0];
                force_m[(column+1)*2*maxrtlh*3+4] += force_projection2[1];
                force_m[(column+1)*2*maxrtlh*3+5] += force_projection2[2];

                torque_due_to_force_cuda(force_projection1, r_s, r_tmp2, a_tmp2, b_tmp2, c_tmp2, torque1);
                torque_due_to_force_cuda(force_projection2, r_f, r_tmp1, a_tmp1, b_tmp1, c_tmp1, torque2);

		torque_m[column*2*maxrtlh*3] += torque1[0];
                torque_m[column*2*maxrtlh*3+1] += torque1[1];
                torque_m[column*2*maxrtlh*3+2] += torque1[2];

		torque_m[(column+1)*2*maxrtlh*3+3] += torque2[0];
                torque_m[(column+1)*2*maxrtlh*3+4] += torque2[1];
                torque_m[(column+1)*2*maxrtlh*3+5] += torque2[2];

        }

	__threadfence();

	//Twisting

	if (column < n-1){

                ic = 0;
                for (count = 0; count <= column; count++){
                        if(type[count]==1) ic=ic+1;
                }
                ic = ic-1;
                ic1 = 3*ic;
                ic2 = ic1+1;
                ic3 = ic2+1;

                for (int xi = 0; xi <3; xi++){
                        r_tmp1[xi] = r[column*3+3+xi];
                        r_tmp2[xi] = r[column*3+xi];
                        a_tmp1[xi] = a[column*3+3+xi];
                        a_tmp2[xi] = a[column*3+xi];
                        b_tmp1[xi] = b[column*3+3+xi];
                        b_tmp2[xi] = b[column*3+xi];
                        c_tmp1[xi] = c[column*3+3+xi];
                        c_tmp2[xi] = c[column*3+xi];
                }

                first_coord_cuda(type[column+1], r_tmp1, a_tmp1, b_tmp1, c_tmp1, r_f);
                second_coord_cuda(type[column], r_tmp2, a_tmp2, b_tmp2, c_tmp2, r_s);

                if (type[column]==0){
                        alpha_b = alpha[column-1];
                        beta_b = beta[column-1];
                        gamma_b = gamma[column-1];
                        phi_o_b = phi_o[column-1];
                        gamma_n = gamma[column-1];
                        for (int xi = 0; xi <3; xi++){
                                b_o[xi] = b[column*3+xi];
                                c_o[xi] = c[column*3+xi];
                        }
                }else{
                        alpha_b = alpha[column];
                        beta_b = beta_p[ic];
                        gamma_b = gamma[column];
                        phi_o_b = phi_o[column];
                        gamma_n = gamma_p[ic];
                        for (int xi = 0; xi <3; xi++){
                                b_o[xi] = b_dna[ic1+xi];
                                c_o[xi] = c_dna[ic1+xi];
                        }
                }

                Twisting_force_projection_cuda(s, alpha[column], beta[column], gamma[column], phi_o[column],  length[column], alpha_b, beta_b, gamma_b, phi_o_b, gamma_n, b_o, c_o, force_projection1, force_projection2, Energy_m[column]);

		force_m[column*2*maxrtlh*3] += force_projection1[0];
                force_m[column*2*maxrtlh*3+1] += force_projection1[1];
                force_m[column*2*maxrtlh*3+2] += force_projection1[2];

		force_m[(column+1)*2*maxrtlh*3+3] += force_projection2[0];
                force_m[(column+1)*2*maxrtlh*3+4] += force_projection2[1];
                force_m[(column+1)*2*maxrtlh*3+5] += force_projection2[2];

                torque_due_to_force_cuda(force_projection1, r_s, r_tmp2, a_tmp2, b_tmp2, c_tmp2, torque1);
                torque_due_to_force_cuda(force_projection2, r_f, r_tmp1, a_tmp1, b_tmp1, c_tmp1, torque2);

		torque_m[column*2*maxrtlh*3] += torque1[0];
                torque_m[column*2*maxrtlh*3+1] += torque1[1];
                torque_m[column*2*maxrtlh*3+2] += torque1[2];

		torque_m[(column+1)*2*maxrtlh*3+3] += torque2[0];
                torque_m[(column+1)*2*maxrtlh*3+4] += torque2[1];
                torque_m[(column+1)*2*maxrtlh*3+5] += torque2[2];

        }

	__threadfence();

	//Mechanical Torques

        if (column < n-1){

                im1 = column-1;
                i1 = column*3;
                i2 = i1+1;
                i3 = i2+1;
                ib1 = i1-3;
                ib2 = ib1+1;
                ib3 = ib2+1;

                ic = 0;
                for (count = 0; count <= column; count++){
                        if(type[count]==1) ic=ic+1;
                }
                ic = ic-1;
                ic1 = 3*ic;
                ic2 = ic1+1;
                ic3 = ic2+1;

                if (type[column]==0){
                        torque[i1] = s*(alpha[column]+gamma[column]+phi_o[column]-alpha[im1]-gamma[im1]-phi_o[im1]);
                        torque[i2] = 0.0;
                        torque[i3] = 0.0;
                }else{
                        ada = a_dna[ic1]*a[i1] + a_dna[ic2]*a[i2] + a_dna[ic3]*a[i3];
                        adb = a_dna[ic1]*b[i1] + a_dna[ic2]*b[i2] + a_dna[ic3]*b[i3];
                        adc = a_dna[ic1]*c[i1] + a_dna[ic2]*c[i2] + a_dna[ic3]*c[i3];
                        
                        mag = s*(alpha[column]+gamma[column]-phi_o[column]);
                        torque[i1] = torque[i1] + mag*ada;
                        torque[i2] = torque[i2] + mag*adb;
                        torque[i3] = torque[i3] + mag*adc;
                        if (column > 0){
                                mag = -s*(alpha[im1]+gamma[im1]-phi_o[im1]);
                                torque[i1] = torque[i1] + mag*co;
                                torque[i2] = torque[i2] + mag*si;
                                torque[i3] = torque[i3] + 0.0;
                        }

                        //Extra Bending torque

                        torque[i2] = torque[i2] - g*beta_p[ic]*adc/sin(beta_p[ic]);
                        torque[i3] = torque[i3] + g*beta_p[ic]*adb/sin(beta_p[ic]);
                
                        if (column > 0){
                                ada = a[ib1]*a[i1] + a[ib2]*a[i2] + a[ib3]*a[i3];
                                adb = a[ib1]*b[i1] + a[ib2]*b[i2] + a[ib3]*b[i3];
                                adc = a[ib1]*c[i1] + a[ib2]*c[i2] + a[ib3]*c[i3];
                                
                                torque[i1] = torque[i1] + g*beta[im1]*(si*adc)/sin(beta[im1]);
                                torque[i2] = torque[i2] - g*beta[im1]*(co*adc)/sin(beta[im1]);
                                torque[i3] = torque[i3] + g*beta[im1]*(co*adb-si*ada)/sin(beta[im1]);
                        }

                        //Extra Twisting torque

                        s1 = sin(alpha_p[ic]);
                        c1 = cos(alpha_p[ic]);
                        mag = s*(alpha[column]+gamma[column]-phi_o[column])*tan(0.5*beta_p[ic]);
                        cda = c_dna[ic1]*a[i1] + c_dna[ic2]*a[i2] + c_dna[ic3]*a[i3];
                        bda = b_dna[ic1]*a[i1] + b_dna[ic2]*a[i2] + b_dna[ic3]*a[i3];
                        cdb = c_dna[ic1]*b[i1] + c_dna[ic2]*b[i2] + c_dna[ic3]*b[i3];
                        bdb = b_dna[ic1]*b[i1] + b_dna[ic2]*b[i2] + b_dna[ic3]*b[i3];
                        cdc = c_dna[ic1]*c[i1] + c_dna[ic2]*c[i2] + c_dna[ic3]*c[i3];
                        bdc = b_dna[ic1]*c[i1] + b_dna[ic2]*c[i2] + b_dna[ic3]*c[i3];

                        torque[i1] = torque[i1] - mag*(s1*cda + c1*bda);
                        torque[i2] = torque[i2] - mag*(s1*cdb + c1*bdb);
                        torque[i3] = torque[i3] - mag*(s1*cdc + c1*bdc);

                        if (column > 0){
                                s1 = sin(gamma[im1]);
                                c1 = cos(gamma[im1]);
                                mag = s*(alpha[im1]+gamma[im1]-phi_o[im1])*tan(0.5*beta[im1]);
                                cda = 0.0;
                                bda = -si;
                                cdb = 0.0;
                                bdb = co;
                                cdc = 1.0;
                                bdc = 0.0;
                                
                                torque[i1] = torque[i1] - mag*(s1*cda - c1*bda);
                                torque[i2] = torque[i2] - mag*(s1*cdb - c1*bdb);
                                torque[i3] = torque[i3] - mag*(s1*cdc - c1*bdc);
                        }

                }
        
        }

	__threadfence();

	//Additional torque for last bead

        if (column==0){

                torque[n3-3] = -s*(alpha[n-2]+gamma[n-2]-phi_o[n-2]);
                torque[n3-2] = 0.0;
                torque[n3-1] = 0.0;

        }


}

__global__ void init_force_torque_m(int maxrtlh, double* force_m, double* torque_m, double* t_force_m, double* LH_force_m, double* temp_force_m, double* Energy_m){

	// Calculate Row and Column
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < maxrtlh*2 && j < maxrtlh*3){
                force_m[i*maxrtlh*3+j] = 0.0;
                torque_m[i*maxrtlh*3+j] = 0.0;
		t_force_m[i*maxrtlh*3+j] = 0.0;
		LH_force_m[i*maxrtlh*3+j] = 0.0;
		temp_force_m[i*maxrtlh*3+j] = 0.0;
        }

	if (i < maxrtlh && j< maxrtlh){
                Energy_m[i*maxrtlh+j]=0.0;
        }

}


__global__ void Electrostatic_and_Excluded_volume_force_cuda(int maxrtlh, int n, int n3, int n_c, int nc3, int* type, double* r, double* a, double* b, double* c, double debyell, double debye, double q_l, double k_e, double k_ex, double k_h1, double sigma_DNA_DNA, double sigma_DNA_Core, double sigma_Core_Core, int Nq, int Nq3, double* core_pos, double* core_q, double* force, double* torque, double* Energy, double* force_m, double* torque_m, double* Energy_m){

	double ql_ql, dist;
        int k, l, ch;
        int i1, i2, i3, j1, j2, j3, k1, k2, k3, l1, l2, l3;
        double mi, Rcut, temp, temp1, temp2;
        double z[3];
        double fa, fb, fc;
        double g1, s1, s2;
	

	Rcut = 25.0;
        ql_ql = q_l*q_l;

	// Calculate Row and Column
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n-1){
                i1 = 3*i;
                i2 = i1+1;
                i3 = i2+1;

		if (j >=  i+1 and j < n){
                        j1 = j*3;
                        j2 = j1+1;
                        j3 = j2+1;

                        dist = (r[j1]-r[i1])*(r[j1]-r[i1])+(r[j2]-r[i2])*(r[j2]-r[i2])+(r[j3]-r[i3])*(r[j3]-r[i3]);
			dist = sqrt(dist);



                        ch = 1;
                        if (dist > Rcut) ch=0;
                        if (type[i+1] != 0) ch=0;
                        if (i > 0 and type[i-1] != 0) ch=0;
                        if (j < n-1){
                                if (type[j+1] != 0) ch=0;
                        }
                        if (type[j-1] != 0) ch=0;


			if (ch == 1){
                                if (type[i] == 0){
                                        if (type[j] == 0){
						if (abs(i-j) > 1){
							mi = 1.0/dist;
                                                        z[0] = mi*(r[i1]-r[j1]);
                                                        z[1] = mi*(r[i2]-r[j2]);
                                                        z[2] = mi*(r[i3]-r[j3]);

                                                        temp = -debyell*dist;

                                                        g1 = k_e*ql_ql*exp(temp)*(debyell*dist+1.0)/(dist*dist);

							Energy_m[i*maxrtlh+j] = Energy_m[i*maxrtlh+j] + (k_e*ql_ql/dist)*exp(temp);
							if (dist <= 8){
                                                                s1 = sigma_DNA_DNA;
                                                                s2 = sigma_DNA_DNA;
                                                                temp1 = s1/dist;
                                                                temp2 = s2/dist;
                                                                g1 = g1 + k_ex*((12.0/s1)*pow(temp1,13)-(6.0/s2)*pow(temp2,7));
                                                        	Energy_m[i*maxrtlh+j] = Energy_m[i*maxrtlh+j] + k_ex*(pow(temp1,12) - pow(temp2,6));
                                                        }

							force_m[i*2*maxrtlh*3+j*3] += g1*z[0];
							force_m[i*2*maxrtlh*3+j*3+1] += g1*z[1];
							force_m[i*2*maxrtlh*3+j*3+2] += g1*z[2];
							force_m[j*2*maxrtlh*3+maxrtlh*3+i*3] -= g1*z[0];
							force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] -= g1*z[1];
							force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] -= g1*z[2];

						}
					}else{
						//DNA-Core interaction

                                                for (k=0;k<Nq;k++){
                                                        k1 = 3*k;
                                                        k2 = k1+1;
                                                        k3 = k2+1;
                                                        z[0] = (r[i1]-(r[j1]+a[j1]*core_pos[k1]+b[j1]*core_pos[k2]+c[j1]*core_pos[k3]));
                                                        z[1] = (r[i2]-(r[j2]+a[j2]*core_pos[k1]+b[j2]*core_pos[k2]+c[j2]*core_pos[k3]));
                                                        z[2] = (r[i3]-(r[j3]+a[j3]*core_pos[k1]+b[j3]*core_pos[k2]+c[j3]*core_pos[k3]));
                                                        dist = z[0]*z[0]+z[1]*z[1]+z[2]*z[2];
							dist = sqrt(dist);
                                                        mi = 1.0/dist;
                                                        z[0] = mi*z[0];
                                                        z[1] = mi*z[1];
                                                        z[2] = mi*z[2];

                                                        if (abs(i-j)>1){

								temp = -debyell*dist;
                                                                g1 = k_e*q_l*core_q[k]*exp(temp)*(debye*dist+1.0)/(dist*dist);
                                                        	Energy_m[i*maxrtlh+j] = Energy_m[i*maxrtlh+j] + (k_e*q_l*core_q[k]/dist)*exp(temp);

								force_m[i*2*maxrtlh*3+j*3] += g1*z[0];
                                	                        force_m[i*2*maxrtlh*3+j*3+1] += g1*z[1];
                        	                                force_m[i*2*maxrtlh*3+j*3+2] += g1*z[2];
                	                                        force_m[j*2*maxrtlh*3+maxrtlh*3+i*3] -= g1*z[0];
        	                                                force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] -= g1*z[1];
	                                                        force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] -= g1*z[2];

								//torque due to force
                                                                fa = -g1*(a[j1]*z[0]+a[j2]*z[1]+a[j3]*z[2]);
                                                                fb = -g1*(b[j1]*z[0]+b[j2]*z[1]+b[j3]*z[2]);
								fc = -g1*(c[j1]*z[0]+c[j2]*z[1]+c[j3]*z[2]);

								torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3] = torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3] + fc*core_pos[k2] - fb*core_pos[k3];
                                                                torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] = torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] + fa*core_pos[k3] - fc*core_pos[k1];
                                                                torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] = torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] + fb*core_pos[k1] - fa*core_pos[k2];



							}

							// Excluded Volume force

	                                                if (dist <= 8.0 and core_q[k]>0){
        	                                                s1 = sigma_DNA_Core;
                	                                        s2 = sigma_DNA_Core;
								temp1 = s1/dist;
	                                                        temp2 = s2/dist;
                       	                                        g1 = k_ex*((12.0/s1)*pow((temp1),13)-(6.0/s2)*pow((temp2),7));
								Energy_m[i*maxrtlh+j] = Energy_m[i*maxrtlh+j] + k_ex*(pow(temp1,12) - pow(temp2,6));


								force_m[i*2*maxrtlh*3+j*3] += g1*z[0];
                                                                force_m[i*2*maxrtlh*3+j*3+1] += g1*z[1];
                                                                force_m[i*2*maxrtlh*3+j*3+2] += g1*z[2];
                                                                force_m[j*2*maxrtlh*3+maxrtlh*3+i*3] -= g1*z[0];
                                                                force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] -= g1*z[1];
								force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] -= g1*z[2];

	                                                        //torque due to force
        	                                                fa = -g1*(a[j1]*z[0]+a[j2]*z[1]+a[j3]*z[2]);
               	                                                fb = -g1*(b[j1]*z[0]+b[j2]*z[1]+b[j3]*z[2]);
                       	                                        fc = -g1*(c[j1]*z[0]+c[j2]*z[1]+c[j3]*z[2]);

								torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3] = torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3] + fc*core_pos[k2] - fb*core_pos[k3];
                                                                torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] = torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] + fa*core_pos[k3] - fc*core_pos[k1];
                                                                torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] = torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] + fb*core_pos[k1] - fa*core_pos[k2];

                                                        }
							
						}
					}
				
				}else{

                                        //Core-DNA interaction
                                        if (type[j] == 0){
                                                for (k=0;k<Nq;k++){
                                                        k1 = 3*k;
                                                        k2 = k1+1;
                                                        k3 = k2+1;
                                                        z[0] = (-r[j1]+(r[i1]+a[i1]*core_pos[k1]+b[i1]*core_pos[k2]+c[i1]*core_pos[k3]));
                                                        z[1] = (-r[j2]+(r[i2]+a[i2]*core_pos[k1]+b[i2]*core_pos[k2]+c[i2]*core_pos[k3]));
                                                        z[2] = (-r[j3]+(r[i3]+a[i3]*core_pos[k1]+b[i3]*core_pos[k2]+c[i3]*core_pos[k3]));
                                                        dist = z[0]*z[0]+z[1]*z[1]+z[2]*z[2];
                                                        dist = sqrt(dist);

                                                        mi = 1.0/dist;
                                                        z[0] = mi*z[0];
                                                        z[1] = mi*z[1];
                                                        z[2] = mi*z[2];
                                                        if (abs(i-j) > 1){
                                                                temp = -debye*dist;
                                                                g1 = k_e*q_l*core_q[k]*exp(temp)*(debye*dist+1.0)/(dist*dist);
								Energy_m[i*maxrtlh+j] = Energy_m[i*maxrtlh+j] + (k_e*q_l*core_q[k]/dist)*exp(temp);

								force_m[i*2*maxrtlh*3+j*3] += g1*z[0];
                                                                force_m[i*2*maxrtlh*3+j*3+1] += g1*z[1];
                                                                force_m[i*2*maxrtlh*3+j*3+2] += g1*z[2];
                                                                force_m[j*2*maxrtlh*3+maxrtlh*3+i*3] -= g1*z[0];
                                                                force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] -= g1*z[1];
                                                                force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] -= g1*z[2];

                                                                //torque due to force
                                                                fa = g1*(a[i1]*z[0]+a[i2]*z[1]+a[i3]*z[2]);
                                                                fb = g1*(b[i1]*z[0]+b[i2]*z[1]+b[i3]*z[2]);
                                                                fc = g1*(c[i1]*z[0]+c[i2]*z[1]+c[i3]*z[2]);
								
								torque_m[i*2*maxrtlh*3+j*3] = torque_m[i*2*maxrtlh*3+j*3] + fc*core_pos[k2] - fb*core_pos[k3];
                                                                torque_m[i*2*maxrtlh*3+j*3+1] = torque_m[i*2*maxrtlh*3+j*3+1] + fa*core_pos[k3] - fc*core_pos[k1];
                                                                torque_m[i*2*maxrtlh*3+j*3+2] = torque_m[i*2*maxrtlh*3+j*3+2] + fb*core_pos[k1] - fa*core_pos[k2];

							}

                                                        //Excluded Volume force
                                                        if (dist <= 8.0 and core_q[k]>0){
                                                                s1 = sigma_DNA_Core;
                                                                s2 = sigma_DNA_Core;
                                                                temp1 = s1/dist;
                                                                temp2 = s2/dist;

								Energy_m[i*maxrtlh+j] = Energy_m[i*maxrtlh+j] + k_ex*(pow(temp1,12) - pow(temp2,6));
                                                                g1 = k_ex*((12.0/s1)*pow(temp1,13)-(6.0/s2)*pow(temp2,7));

								force_m[i*2*maxrtlh*3+j*3] += g1*z[0];
                                                                force_m[i*2*maxrtlh*3+j*3+1] += g1*z[1];
                                                                force_m[i*2*maxrtlh*3+j*3+2] += g1*z[2];
                                                                force_m[j*2*maxrtlh*3+maxrtlh*3+i*3] -= g1*z[0];
                                                                force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] -= g1*z[1];
                                                                force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] -= g1*z[2];
                                                                
								//torque due to force
                                                                fa = g1*(a[i1]*z[0]+a[i2]*z[1]+a[i3]*z[2]);
                                                                fb = g1*(b[i1]*z[0]+b[i2]*z[1]+b[i3]*z[2]);
                                                                fc = g1*(c[i1]*z[0]+c[i2]*z[1]+c[i3]*z[2]);

								torque_m[i*2*maxrtlh*3+j*3] = torque_m[i*2*maxrtlh*3+j*3] + fc*core_pos[k2] - fb*core_pos[k3];
                                                                torque_m[i*2*maxrtlh*3+j*3+1] = torque_m[i*2*maxrtlh*3+j*3+1] + fa*core_pos[k3] - fc*core_pos[k1];
                                                                torque_m[i*2*maxrtlh*3+j*3+2] = torque_m[i*2*maxrtlh*3+j*3+2] + fb*core_pos[k1] - fa*core_pos[k2];

							}
                                                }
					}else{//Core-Core interaction

                                                for (k = 0; k < Nq; k++){
                                                        k1 = 3*k;
                                                        k2 = k1+1;
                                                        k3 = k2+1;
                                                        for (l = 0; l < Nq; l++){
                                                                l1 = 3*l;
                                                                l2 = l1+1;
                                                                l3 = l2+1;
                                                                z[0] = (r[i1]+a[i1]*core_pos[k1]+b[i1]*core_pos[k2]+c[i1]*core_pos[k3] - (r[j1]+a[j1]*core_pos[l1]+b[j1]*core_pos[l2]+c[j1]*core_pos[l3]));
                                                                z[1] = (r[i2]+a[i2]*core_pos[k1]+b[i2]*core_pos[k2]+c[i2]*core_pos[k3] - (r[j2]+a[j2]*core_pos[l1]+b[j2]*core_pos[l2]+c[j2]*core_pos[l3]));
                                                                z[2] = (r[i3]+a[i3]*core_pos[k1]+b[i3]*core_pos[k2]+c[i3]*core_pos[k3] - (r[j3]+a[j3]*core_pos[l1]+b[j3]*core_pos[l2]+c[j3]*core_pos[l3]));
                                                                dist = z[0]*z[0]+z[1]*z[1]+z[2]*z[2];
                                                                dist = sqrt(dist);

                                                                mi = 1.0/dist;
                                                                z[0] = mi*z[0];
                                                                z[1] = mi*z[1];
                                                                z[2] = mi*z[2];

                                                                temp = -debye*dist;
                                                                g1 = k_e*core_q[k]*core_q[l]*exp(temp)*(debye*dist+1.0)/(dist*dist);
								Energy_m[i*maxrtlh+j] = Energy_m[i*maxrtlh+j] + (k_e*core_q[l]*core_q[k]/dist)*exp(temp);

								force_m[i*2*maxrtlh*3+j*3] += g1*z[0];
                                                                force_m[i*2*maxrtlh*3+j*3+1] += g1*z[1];
                                                                force_m[i*2*maxrtlh*3+j*3+2] += g1*z[2];
                                                                force_m[j*2*maxrtlh*3+maxrtlh*3+i*3] -= g1*z[0];
                                                                force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] -= g1*z[1];
                                                                force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] -= g1*z[2];
  
                                                                //torque due to force
                                                                fa = g1*(a[i1]*z[0]+a[i2]*z[1]+a[i3]*z[2]);
                                                                fb = g1*(b[i1]*z[0]+b[i2]*z[1]+b[i3]*z[2]);
								fc = g1*(c[i1]*z[0]+c[i2]*z[1]+c[i3]*z[2]);

								torque_m[i*2*maxrtlh*3+j*3] = torque_m[i*2*maxrtlh*3+j*3] + fc*core_pos[k2] - fb*core_pos[k3];
                                                                torque_m[i*2*maxrtlh*3+j*3+1] = torque_m[i*2*maxrtlh*3+j*3+1] + fa*core_pos[k3] - fc*core_pos[k1];
                                                                torque_m[i*2*maxrtlh*3+j*3+2] = torque_m[i*2*maxrtlh*3+j*3+2] + fb*core_pos[k1] - fa*core_pos[k2];

								fa = -g1*(a[j1]*z[0]+a[j2]*z[1]+a[j3]*z[2]);
                                                                fb = -g1*(b[j1]*z[0]+b[j2]*z[1]+b[j3]*z[2]);
                                                                fc = -g1*(c[j1]*z[0]+c[j2]*z[1]+c[j3]*z[2]);

								torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3] = torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3] + fc*core_pos[l2] - fb*core_pos[l3];
                                                                torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] = torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] + fa*core_pos[l3] - fc*core_pos[l1];
                                                                torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] = torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] + fb*core_pos[l1] - fa*core_pos[l2];

                                                                //Excluded Volume force
                                                                if (dist <= 8.0){
                                                                        s1 = sigma_Core_Core;
                                                                        s2 = sigma_Core_Core;
                                                                        temp1 = s1/dist;
                                                                        temp2 = s2/dist;
                                                                        g1 = k_ex*((12.0/s1)*pow(temp1,13)-(6.0/s2)*pow(temp2,7));

									Energy_m[i*maxrtlh+j] = Energy_m[i*maxrtlh+j] + k_ex*(pow(temp1,12) - pow(temp2,6));

									force_m[i*2*maxrtlh*3+j*3] += g1*z[0];
                                                                	force_m[i*2*maxrtlh*3+j*3+1] += g1*z[1];
	                                                                force_m[i*2*maxrtlh*3+j*3+2] += g1*z[2];
        	                                                        force_m[j*2*maxrtlh*3+maxrtlh*3+i*3] -= g1*z[0];
                	                                                force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] -= g1*z[1];
                        	                                        force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] -= g1*z[2];


                                                                        //torque due to force
                                                                        fa = g1*(a[i1]*z[0]+a[i2]*z[1]+a[i3]*z[2]);
                                                                        fb = g1*(b[i1]*z[0]+b[i2]*z[1]+b[i3]*z[2]);
                                                                        fc = g1*(c[i1]*z[0]+c[i2]*z[1]+c[i3]*z[2]);

									torque_m[i*2*maxrtlh*3+j*3] = torque_m[i*2*maxrtlh*3+j*3] + fc*core_pos[k2] - fb*core_pos[k3];
        	                                                        torque_m[i*2*maxrtlh*3+j*3+1] = torque_m[i*2*maxrtlh*3+j*3+1] + fa*core_pos[k3] - fc*core_pos[k1];
	                                                                torque_m[i*2*maxrtlh*3+j*3+2] = torque_m[i*2*maxrtlh*3+j*3+2] + fb*core_pos[k1] - fa*core_pos[k2];

									fa = -g1*(a[j1]*z[0]+a[j2]*z[1]+a[j3]*z[2]);
                                                                        fb = -g1*(b[j1]*z[0]+b[j2]*z[1]+b[j3]*z[2]);
                                                                        fc = -g1*(c[j1]*z[0]+c[j2]*z[1]+c[j3]*z[2]);
									
									torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3] = torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3] + fc*core_pos[l2] - fb*core_pos[l3];
        	                                                        torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] = torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] + fa*core_pos[l3] - fc*core_pos[l1];
  	     	                                                        torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] = torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] + fb*core_pos[l1] - fa*core_pos[l2];

								}

                                                        }
                                                }

                                        }
				}
					
			}


                        }



                


        }
}

__global__ void step_init(int n, int n_tail, int n_LH, double* force, double* force_n, double* torque, double* torque_n, double* t_force, double* t_force_n, double* LH_force, double* LH_force_n, double* Energy, double* r_all, double* r, double* r_t, double* r_lh){

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	Energy[0] = 0.0;
	if (i < n*3){
		force[i]=0;
                force_n[i]=0;
                torque[i]=0;
                torque_n[i]=0;
	}
	if (i < n_tail*3){
		t_force[i] = 0;
		t_force_n[i] = 0;
	}
	if (i < n_LH*3){
		LH_force[i] = 0;
		LH_force_n[i] = 0;
	}

	if (i<n){
		r_all[i*3] = r[i*3];
		r_all[i*3+1] = r[i*3+1];
		r_all[i*3+2] = r[i*3+2];
	}else if (i<n+n_tail){
		r_all[i*3] = r_t[i*3-n*3];
                r_all[i*3+1] = r_t[i*3-n*3+1];
                r_all[i*3+2] = r_t[i*3-n*3+2];
	}else if (i<n+n_tail+n_LH){
		r_all[i*3] = r_lh[i*3-n*3-n_tail*3];
                r_all[i*3+1] = r_lh[i*3-n*3-n_tail*3+1];
                r_all[i*3+2] = r_lh[i*3-n*3-n_tail*3+2];
	}

}

__global__ void force_reduction(int n, int n_tail, int n_LH, int maxrtlh, int maxrtlh3, double* force_m, double* force, double* torque_m, double* torque, double* t_force_m, double* t_force, double* LH_force_m, double* LH_force){

        int column = blockIdx.x * blockDim.x + threadIdx.x;
        int index1, index2;

        if (column < maxrtlh){
                for (int i=0; i<maxrtlh; i++){
                        index1 = column*2*maxrtlh3+i*3;
                        index2 = column*2*maxrtlh3+maxrtlh3+i*3;
			if (column<n){
				force[column*3] += force_m[index1] + force_m[index2];
                        	force[column*3+1] += force_m[index1+1] + force_m[index2+1];
                        	force[column*3+2] += force_m[index1+2] + force_m[index2+2];
			}
			if (column<n){
				torque[column*3] += torque_m[index1] + torque_m[index2];
                        	torque[column*3+1] += torque_m[index1+1] + torque_m[index2+1];
                        	torque[column*3+2] += torque_m[index1+2] + torque_m[index2+2];
			}
			if (column<n_tail){
				t_force[column*3] += t_force_m[index1] + t_force_m[index2];
                        	t_force[column*3+1] += t_force_m[index1+1] + t_force_m[index2+1];
                        	t_force[column*3+2] += t_force_m[index1+2] + t_force_m[index2+2];
			}
			if (column< n_LH){
				LH_force[column*3] += LH_force_m[index1] + LH_force_m[index2];
                       		LH_force[column*3+1] += LH_force_m[index1+1] + LH_force_m[index2+1];
                        	LH_force[column*3+2] += LH_force_m[index1+2] + LH_force_m[index2+2];
			}
                }
        }

}

__global__ void Energy_reduction(int maxrtlh, double* Energy, double* Energy_m){
        int column = blockIdx.x * blockDim.x + threadIdx.x;

        if (column< maxrtlh){
                for (int i=1; i<maxrtlh; i++){
                        Energy_m[column] += Energy_m[i*maxrtlh+column];
                }
        }

        __threadfence();

        if (column==0){
                for (int i=0; i<maxrtlh; i++){
                        Energy[0] += Energy_m[i];
                }
        }


}

__global__ void update_tail_beta_cuda(int n_tail, double* r_t, double* beta_t, int* t_grp){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
        double x12, y12, z12, r12, x32, y32, z32, r32, p123;

        if (i < n_tail){
                if((t_grp[i] == t_grp[i+1]) and (t_grp[i] == t_grp[i+2]) and (i < n_tail-2)){
                        x12 = r_t[i*3] - r_t[i*3+3];
                        y12 = r_t[i*3+1] - r_t[i*3+4];
                        z12 = r_t[i*3+2] - r_t[i*3+5];

                        r12 = x12*x12 + y12*y12 + z12*z12;
			r12 = sqrt(r12);

                        x32 = r_t[i*3+6] - r_t[i*3+3];
                        y32 = r_t[i*3+7] - r_t[i*3+4];
                        z32 = r_t[i*3+8] - r_t[i*3+5];

                        r32 = x32*x32 + y32*y32 + z32*z32;
			r32 = sqrt(r32);

                        p123 = (x12*x32 + y12*y32 + z12*z32)/(r12*r32);

                        beta_t[i] = acos(p123);
                }else{
                        beta_t[i] = 0.0;
                }

        }

}


__global__ void tail_force_cuda_str_ben(int maxrtlh, int n, int n_t, int* type, int n_tail, int n_tail3, double* tail_pos, int* tail_fix, int* nc_t_flag, double* r_t, double* beta_t, double* h_t, double* g_t, double* lo_t, double* beta_o_t, double h, double k_e, double debye, double k_ex, double* t_q, double* t_rad, int* t_grp, int* t_fix, double* r, double* a, double* b, double* c, double q_l, int Nq, int Nq3, double* core_pos, double* core_q, double* t_force, double* force, double* torque, double sigma_Tail_Tail, double sigma_Tail_Linker, double sigma_Tail_Core, double* Energy, double* t_force_m, double* Energy_m){

        double force_projection1[3], force_projection2[3],force_projection3[3];
        double r_t_tmp1[3], r_t_tmp2[3], r_t_tmp3[3];

	// Calculate Row and Column
        int i = blockIdx.x * blockDim.x + threadIdx.x;

	//tail stretching
	if (i<n_tail-1){
                if (t_grp[i]==t_grp[i+1]){
                        for (int ix = 0; ix<3; ix++){
                                r_t_tmp1[ix] = r_t[i*3+ix];
                                r_t_tmp2[ix] = r_t[i*3+3+ix];
                        }
                        Force_Stretching_cuda(h_t[i], r_t_tmp1, r_t_tmp2, lo_t[i], force_projection1, force_projection2, Energy_m[i]);

                        t_force_m[i*2*maxrtlh*3] += force_projection1[0];
                        t_force_m[i*2*maxrtlh*3+1] += force_projection1[1];
                        t_force_m[i*2*maxrtlh*3+2] += force_projection1[2];

			t_force_m[(i+1)*2*maxrtlh*3+3] += force_projection2[0];
                        t_force_m[(i+1)*2*maxrtlh*3+4] += force_projection2[1];
                        t_force_m[(i+1)*2*maxrtlh*3+5] += force_projection2[2];

                }
        }

	__threadfence();
	
	//tail bending
	if (i<n_tail-2){
                if ((t_grp[i]==t_grp[i+1]) and (t_grp[i]==t_grp[i+2])){
                        for (int ix = 0; ix<3; ix++){
                                r_t_tmp1[ix] = r_t[i*3+ix];
                                r_t_tmp2[ix] = r_t[i*3+3+ix];
                                r_t_tmp3[ix] = r_t[i*3+6+ix];
                        }
                        Force_Bending_cuda(g_t[i],beta_t[i],beta_o_t[i], r_t_tmp1, r_t_tmp2, r_t_tmp3, force_projection1, force_projection2, force_projection3, Energy_m[i]);

			t_force_m[i*2*maxrtlh*3] -= force_projection1[0];
                        t_force_m[i*2*maxrtlh*3+1] -= force_projection1[1];
                        t_force_m[i*2*maxrtlh*3+2] -= force_projection1[2];

			t_force_m[(i+1)*2*maxrtlh*3+3] -= force_projection2[0];
                        t_force_m[(i+1)*2*maxrtlh*3+4] -= force_projection2[1];
                        t_force_m[(i+1)*2*maxrtlh*3+5] -= force_projection2[2];

			t_force_m[(i+2)*2*maxrtlh*3+6] -= force_projection3[0];
                        t_force_m[(i+2)*2*maxrtlh*3+7] -= force_projection3[1];
                        t_force_m[(i+2)*2*maxrtlh*3+8] -= force_projection3[2];

                }
        }

	__threadfence();

}

__global__ void tail_force_cuda_ele_vdw(int maxrtlh, int n, int n_t, int* type, int n_tail, int n_tail3, double* tail_pos, int* tail_fix, int* nc_t_flag, double* r_t, double* beta_t, double* h_t, double* g_t, double* lo_t, double* beta_o_t, double h, double k_e, double debye, double k_ex, double* t_q, double* t_rad, int* t_grp, int* t_fix, double* r, double* a, double* b, double* c, double q_l, int Nq, int Nq3, double* core_pos, double* core_q, double* t_force, double* force, double* torque, double sigma_Tail_Tail, double sigma_Tail_Linker, double sigma_Tail_Core, double* Energy, double* t_force_m, double* force_m, double* torque_m, double* Energy_m){

	double Stri[3];
        int k;
        double r_to[3];
        double force_projection1[3], force_projection2[3];
        double torque_tc[3];

        double r_t_tmp1[3], r_t_tmp2[3], r_tmp[3], tail_pos_tmp[3], a_tmp[3], b_tmp[3], c_tmp[3], h_stri_tmp[3];

        // Calculate Row and Column
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i<n_tail-1){
                if (j>i and j<n_tail){
			
                        if ((t_grp[i]!=t_grp[j]) or (j>i+2)){
                                for (int ix = 0; ix<3; ix++){
                                        r_t_tmp1[ix] = r_t[i*3+ix];
                                        r_t_tmp2[ix] = r_t[j*3+ix];
                                }
                                Force_Electrostatics_cuda(t_q[i], t_q[j], 1/(4*PI*k_e), debye, r_t_tmp1, r_t_tmp2, force_projection1, force_projection2, Energy_m[i*maxrtlh+j]);
				
				t_force_m[i*2*maxrtlh*3+j*3] += force_projection1[0];
                                t_force_m[i*2*maxrtlh*3+j*3+1] += force_projection1[1];
                                t_force_m[i*2*maxrtlh*3+j*3+2] += force_projection1[2];

				
				t_force_m[j*2*maxrtlh*3+maxrtlh*3+i*3] += force_projection2[0];
                                t_force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] += force_projection2[1];
                                t_force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] += force_projection2[2];

			}

                }
        }

	__threadfence();


	if (i<n_tail){
                if (j<n){
			if(type[j]==0){
				
                                for (int ix = 0; ix<3; ix++){
                                        r_t_tmp1[ix] = r_t[i*3+ix];
                                        r_tmp[ix] = r[j*3+ix];
                                }

				Force_Ele_Vdw_cuda(t_q[i], q_l, 1/(4*PI*k_e), debye,k_ex, sigma_Tail_Linker, r_t_tmp1, r_tmp, force_projection1, force_projection2, Energy_m[i*maxrtlh+j]);

                                t_force_m[i*2*maxrtlh*3+j*3] += force_projection1[0];
                                t_force_m[i*2*maxrtlh*3+j*3+1] += force_projection1[1];
                                t_force_m[i*2*maxrtlh*3+j*3+2] += force_projection1[2];

                                force_m[j*2*maxrtlh*3+maxrtlh*3+i*3] += force_projection2[0];
                                force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] += force_projection2[1];
                                force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] += force_projection2[2];

			}

		}

	}
	__threadfence();


	//Stretching between tail and core

        if  (i < n){
                if (type[i]==1){
			k=0;
			for (int cnti=0; cnti<i;cnti++){
				if (type[cnti]==1){
					k=k+1;
				}
			}
                        if(nc_t_flag[k]==1){
                                if (j < n_t){
                                        if (tail_fix[j]==1){

                                                r_to[0] = r[i*3] + a[i*3]*tail_pos[j*3] + b[i*3]*tail_pos[j*3+1] + c[i*3]*tail_pos[j*3+2];
                                                r_to[1] = r[i*3+1] + a[i*3+1]*tail_pos[j*3] + b[i*3+1]*tail_pos[j*3+1] + c[i*3+1]*tail_pos[j*3+2];
                                                r_to[2] = r[i*3+2] + a[i*3+2]*tail_pos[j*3] + b[i*3+2]*tail_pos[j*3+1] + c[i*3+2]*tail_pos[j*3+2];

                                                Stri[0] = r_t[k*n_t*3+j*3] - r_to[0];
                                                Stri[1] = r_t[k*n_t*3+j*3+1] - r_to[1];
                                                Stri[2] = r_t[k*n_t*3+j*3+2] - r_to[2];

                                                Energy_m[i*maxrtlh+j] = Energy_m[i*maxrtlh+j] + h*(Stri[0]*Stri[0]+Stri[1]*Stri[1]+Stri[2]*Stri[2])/2;

                                                t_force_m[(k*n_t+j)*2*maxrtlh*3+i*3] -= h*Stri[0];
                                                t_force_m[(k*n_t+j)*2*maxrtlh*3+i*3+1] -= h*Stri[1];
                                                t_force_m[(k*n_t+j)*2*maxrtlh*3+i*3+2] -= h*Stri[2];

                                                force_m[i*2*maxrtlh*3+j*3] += h*Stri[0];
                                                force_m[i*2*maxrtlh*3+j*3+1] += h*Stri[1];
                                                force_m[i*2*maxrtlh*3+j*3+2] += h*Stri[2];

                                                // torque due to force

                                                for (int ix = 0; ix<3; ix++){
                                                        tail_pos_tmp[ix] =  tail_pos[j*3+ix];
                                                        a_tmp[ix] = a[i*3+ix];
                                                        b_tmp[ix] = b[i*3+ix];
                                                        c_tmp[ix] = c[i*3+ix];
                                                        h_stri_tmp[ix] = h*Stri[ix];
                                                }

                                                torque_due_to_force_relative_cuda(h_stri_tmp, tail_pos_tmp, a_tmp, b_tmp, c_tmp, torque_tc);
                                                torque_m[i*2*maxrtlh*3+j*3] += torque_tc[0];
                                                torque_m[i*2*maxrtlh*3+j*3+1] += torque_tc[1];
                                                torque_m[i*2*maxrtlh*3+j*3+2] += torque_tc[2];

                                        }
                                }
                        }
                }
        }

}

__global__ void update_LH_beta_cuda(int n_LH, int n_lh_c, double* r_lh, double* beta_lh){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
        double x12, y12, z12, r12, x32, y32, z32, r32, p123;

        if (i < n_LH){
                if((i%n_lh_c!=0) and (i%n_lh_c != n_lh_c-1)){
                        x12 = r_lh[i*3-3] - r_lh[i*3];
                        y12 = r_lh[i*3-2] - r_lh[i*3+1];
                        z12 = r_lh[i*3-1] - r_lh[i*3+2];

                        r12 = x12*x12 + y12*y12 + z12*z12;
			r12 = sqrt(r12);

                        x32 = r_lh[i*3+3] - r_lh[i*3];
                        y32 = r_lh[i*3+4] - r_lh[i*3+1];
                        z32 = r_lh[i*3+5] - r_lh[i*3+2];

                        r32 = x32*x32 + y32*y32 + z32*z32;
			r32 = sqrt(r32);

                        p123 = (x12*x32 + y12*y32 + z12*z32)/(r12*r32);

                        beta_lh[i] = acos(p123);
                }else{
                        beta_lh[i] = 0.0;
                }

        }

}

__global__ void Linker_Histone_str_ben_cuda(int maxrtlh, int n, int n_tail, int n_lh_n, int n_lh_g, int n_lh_c, int* t_grp, int* t_fix, int* type, int n_LH, int n_LH3, double* LH_g_pos, int* LH_conn, int* nc_lh_flag, double* beta_lh, double* r_lh, double* LH_q, double k_e, double debye, double k_ex, double* LH_vdw_hh, double* LH_vdw_hc, double* LH_vdw_hl, double* LH_vdw_ht, double* LH_kstr, double* LH_kben, double* LH_streq, double* LH_betaeq, double* r, double* a, double* b, double* c, double q_l, int Nq, int Nq3, double* core_pos, double* core_q, double* r_t, double* t_q, double* t_force, double* LH_force, double* force, double* torque, double* Energy, double* LH_force_m, double* Energy_m){

        int xi;
        int index;
        double force_projection1[3], force_projection2[3],force_projection3[3];

        double r_lh_tmp1[3], r_lh_tmp2[3], r_lh_tmp3[3];

	// Calculate Row and Column
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        //Stretching force between LH beads

        if(i<n_LH){
                index = i%(n_lh_n+n_lh_c);
                if (index < n_lh_n){
                        index = index;
                }else{
                        index = index + n_lh_g;
                }

                if (LH_conn[index]==1){
                        for (xi = 0; xi<3; xi++){
                                r_lh_tmp1[xi] = r_lh[i*3+xi];
                                r_lh_tmp2[xi] = r_lh[i*3+3+xi];
                        }

                        Force_Stretching_cuda(LH_kstr[index], r_lh_tmp1, r_lh_tmp2, LH_streq[index],force_projection1, force_projection2, Energy_m[i]);
                        LH_force_m[i*2*maxrtlh*3] += force_projection1[0];
                        LH_force_m[i*2*maxrtlh*3+1] += force_projection1[1];
                        LH_force_m[i*2*maxrtlh*3+2] += force_projection1[2];

			LH_force_m[(i+1)*2*maxrtlh*3+3] += force_projection2[0];
                        LH_force_m[(i+1)*2*maxrtlh*3+4] += force_projection2[1];
                        LH_force_m[(i+1)*2*maxrtlh*3+5] += force_projection2[2];
                }
        }

	__threadfence();
	
	//Bending between LH beads

        if(i<n_LH){
                index = i%(n_lh_n+n_lh_c);
                if (index < n_lh_n){
                        index = index;
                }else{
                        index = index + n_lh_g;
                }
                if ((i%n_lh_c!=0) and (i%n_lh_c!=n_lh_c-1)){
                        for (xi = 0; xi<3; xi++){
                                r_lh_tmp1[xi] = r_lh[i*3-3+xi];
                                r_lh_tmp2[xi] = r_lh[i*3+xi];
                                r_lh_tmp3[xi] = r_lh[i*3+3+xi];
                        }
                        Force_Bending_cuda(LH_kben[index],beta_lh[i],LH_betaeq[index],r_lh_tmp1, r_lh_tmp2, r_lh_tmp3,force_projection1,force_projection2,force_projection3, Energy_m[i]);
			
			LH_force_m[(i-1)*2*maxrtlh*3] -= force_projection1[0];
                        LH_force_m[(i-1)*2*maxrtlh*3+1] -= force_projection1[1];
                        LH_force_m[(i-1)*2*maxrtlh*3+2] -= force_projection1[2];

			LH_force_m[i*2*maxrtlh*3+3] -= force_projection2[0];
                        LH_force_m[i*2*maxrtlh*3+4] -= force_projection2[1];
                        LH_force_m[i*2*maxrtlh*3+5] -= force_projection2[2];

                        LH_force_m[(i+1)*2*maxrtlh*3+6] -= force_projection3[0];
                        LH_force_m[(i+1)*2*maxrtlh*3+7] -= force_projection3[1];
                        LH_force_m[(i+1)*2*maxrtlh*3+8] -= force_projection3[2];	


                }
        }

	__threadfence();
}

__global__ void temp_force_2_force_m(int n, int maxrtlh, int* t_grp, int* type, double* temp_force_m, double* force_m, double* torque_m){

	int k, index_c;

	// Calculate Row and Column
        int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < maxrtlh){
		k=0;
        	for (int ccnt=0; ccnt<n; ccnt++){
	        	if (k==(t_grp[j]-1)/10 and type[ccnt]==1){
                		index_c = ccnt;
                        	break;
                	}
                	if(type[ccnt]==1){
                        	k=k+1;
                	}
        	}
		for (int i=0; i< maxrtlh; i++){
			force_m[index_c*2*maxrtlh*3+j*3] += temp_force_m[i*2*maxrtlh*3+j*3];
			force_m[index_c*2*maxrtlh*3+j*3+1] += temp_force_m[i*2*maxrtlh*3+j*3+1];
			force_m[index_c*2*maxrtlh*3+j*3+2] += temp_force_m[i*2*maxrtlh*3+j*3+2];
			torque_m[index_c*2*maxrtlh*3+j*3] += temp_force_m[i*2*maxrtlh*3+maxrtlh*3+j*3];
                        torque_m[index_c*2*maxrtlh*3+j*3+1] += temp_force_m[i*2*maxrtlh*3+maxrtlh*3+j*3+1];
                        torque_m[index_c*2*maxrtlh*3+j*3+2] += temp_force_m[i*2*maxrtlh*3+maxrtlh*3+j*3+2];
		}
	}


}

__global__ void Linker_Histone_ele_vdw_cuda(int maxrtlh, int n, int n_tail, int n_lh_n, int n_lh_g, int n_lh_c, int* t_grp, int* t_fix, int* type, int n_LH, int n_LH3, double* LH_g_pos, int* LH_conn, int* nc_lh_flag, double* beta_lh, double* r_lh, double* LH_q, double k_e, double debye, double k_ex, double* LH_vdw_hh, double* LH_vdw_hc, double* LH_vdw_hl, double* LH_vdw_ht, double* LH_kstr, double* LH_kben, double* LH_streq, double* LH_betaeq, double* r, double* a, double* b, double* c, double q_l, int Nq, int Nq3, double* core_pos, double* core_q, double* r_t, double* t_q, double* t_force, double* LH_force, double* force, double* torque, double* Energy, double* LH_force_m, double* force_m, double* torque_m, double* t_force_m, double* temp_force_m, double* Energy_m){
	
	double Stri[3];
        double distance, fa, fb, fc;
        int k, l, k2, m, m1, xi, count;
        double r_lho[3], r_lho2[3], z[3];
        int index, index1, index2, index_c;
        int nlh = n_lh_n+n_lh_c;
        double torque_hgl[3], torque1[3], torque2[3];
        double force_projection1[3], force_projection2[3];

        double r_lh_tmp1[3], r_lh_tmp2[3], r_t_tmp[3], r_tmp[3], a_tmp[3], b_tmp[3], c_tmp[3], core_pos_tmp[3], LH_g_pos_tmp[3];

	//Electrostatic and Excluded Volume between LH

	// Calculate Row and Column
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        
	if(i<n_LH){
                index1 = i%(n_lh_n+n_lh_c);
                if (index1 < n_lh_n){
                        index1 = index1;
                }else{
                        index1 = index1 + n_lh_g;
                }
                if (j >=i and j<n_LH){
                        index2 = j%(n_lh_n+n_lh_c);
                        if (index2 < n_lh_n){
                                index2 = index2;
                        }else{
                                index2 = index2 + n_lh_g;
                        }

                        if ((i/(n_lh_n+n_lh_c) == j/(n_lh_n+n_lh_c)) and (j > i+1)){

                                for (xi = 0; xi<3; xi++){
                                        r_lh_tmp1[xi] = r_lh[i*3+xi];
                                        r_lh_tmp2[xi] = r_lh[j*3+xi];
                                }

                                Force_Exclude_Volume_cuda(k_ex, (LH_vdw_hh[index1]+LH_vdw_hh[index2])/2, r_lh_tmp1, r_lh_tmp2,force_projection1, force_projection2, Energy_m[i*maxrtlh+j]);

				LH_force_m[i*2*maxrtlh*3+j*3] += force_projection1[0];
                                LH_force_m[i*2*maxrtlh*3+j*3+1] += force_projection1[1];
                                LH_force_m[i*2*maxrtlh*3+j*3+2] += force_projection1[2];

				LH_force_m[j*2*maxrtlh*3+maxrtlh*3+i*3] += force_projection2[0];
                                LH_force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] += force_projection2[1];
                                LH_force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] += force_projection2[2];

                        }else if ((i/(n_lh_n+n_lh_c) != j/(n_lh_n+n_lh_c))){
                                for (xi = 0; xi<3; xi++){
                                        r_lh_tmp1[xi] = r_lh[i*3+xi];
                                        r_lh_tmp2[xi] = r_lh[j*3+xi];
                                }


                                Force_Ele_Vdw_cuda(LH_q[index1], LH_q[index2], 1/(4*PI*k_e), debye,k_ex, (LH_vdw_hh[index1]+LH_vdw_hh[index2])/2, r_lh_tmp1, r_lh_tmp2,force_projection1, force_projection2, Energy_m[i*maxrtlh+j]);

				LH_force_m[i*2*maxrtlh*3+j*3] += force_projection1[0];
                                LH_force_m[i*2*maxrtlh*3+j*3+1] += force_projection1[1];
                                LH_force_m[i*2*maxrtlh*3+j*3+2] += force_projection1[2];
				
				LH_force_m[j*2*maxrtlh*3+maxrtlh*3+i*3] += force_projection2[0];
                                LH_force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] += force_projection2[1];
                                LH_force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] += force_projection2[2];

                        }

                }
        }

	__threadfence();

	//Electrostatic and Excluded Volume between LH and DNA

        if (i<n_LH){
                index = i%(n_lh_n+n_lh_c);
                if (index < n_lh_n){
                        index = index;
                }else{
                        index = index + n_lh_g;
                }
                if (j<n){
                        if (type[j]==0){
                                for (xi = 0; xi<3; xi++){
                                        r_lh_tmp1[xi] = r_lh[i*3+xi];
                                        r_tmp[xi] = r[j*3+xi];
                                }
                                Force_Ele_Vdw_cuda(LH_q[index], q_l, 1/(4*PI*k_e), debye,k_ex, LH_vdw_hl[index], r_lh_tmp1, r_tmp,force_projection1, force_projection2, Energy_m[i*maxrtlh+j]);
				LH_force_m[i*2*maxrtlh*3+j*3] += force_projection1[0];
                                LH_force_m[i*2*maxrtlh*3+j*3+1] += force_projection1[1];
                                LH_force_m[i*2*maxrtlh*3+j*3+2] += force_projection1[2];

				force_m[j*2*maxrtlh*3+i*3] += force_projection2[0];
                                force_m[j*2*maxrtlh*3+i*3+1] += force_projection2[1];
                                force_m[j*2*maxrtlh*3+i*3+2] += force_projection2[2];

                        }
                }
        }

	__threadfence();

	//Electrostatic and Excluded Volume between LH and Tails

        if (i<n_LH){
                index = i%(n_lh_n+n_lh_c);
                if (index < n_lh_n){
                        index = index;
                }else{
                        index = index + n_lh_g;
                }
                if (j<n_tail){
                        for (xi = 0; xi<3; xi++){
                                r_lh_tmp1[xi] = r_lh[i*3+xi];
                                r_t_tmp[xi] = r_t[j*3+xi];
                        }
                        Force_Ele_Vdw_cuda(LH_q[index], t_q[j], 1/(4*PI*k_e), debye,k_ex, LH_vdw_ht[index], r_lh_tmp1, r_t_tmp,force_projection1, force_projection2, Energy_m[i*maxrtlh+j]);
                        LH_force_m[i*2*maxrtlh*3+j*3] += force_projection1[0];
                        LH_force_m[i*2*maxrtlh*3+j*3+1] += force_projection1[1];
                        LH_force_m[i*2*maxrtlh*3+j*3+2] += force_projection1[2];

                        if (t_fix[j]==0){
                                t_force_m[j*2*maxrtlh*3+maxrtlh*3+i*3] += force_projection2[0];
                                t_force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] += force_projection2[1];
                                t_force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] += force_projection2[2];
                        }else{
                                k=0;
                                for (int ccnt=0; ccnt<n; ccnt++){
                                        if (k==(t_grp[j]-1)/10 and type[ccnt]==1){
                                                index_c = ccnt;
                                                break;
                                        }
                                        if(type[ccnt]==1){
                                                k=k+1;
                                        }
                                }

                                for (int ix = 0; ix<3; ix++){
                                        r_tmp[ix] = r[index_c*3+ix];
                                        a_tmp[ix] = a[index_c*3+ix];
                                        b_tmp[ix] = b[index_c*3+ix];
                                        c_tmp[ix] = c[index_c*3+ix];
                                }
                                torque_due_to_force_cuda(force_projection2, r_t_tmp, r_tmp, a_tmp, b_tmp, c_tmp, torque2);
				
				temp_force_m[i*2*maxrtlh*3+j*3] += force_projection2[0];
				temp_force_m[i*2*maxrtlh*3+j*3+1] += force_projection2[1];
				temp_force_m[i*2*maxrtlh*3+j*3+2] += force_projection2[2];
				temp_force_m[i*2*maxrtlh*3+maxrtlh*3+j*3] += torque2[0];
                                temp_force_m[i*2*maxrtlh*3+maxrtlh*3+j*3+1] += torque2[1];
                                temp_force_m[i*2*maxrtlh*3+maxrtlh*3+j*3+2] += torque2[2];
                        }

                }
        }

	__threadfence();

	//Electrostatic and Excluded Volume between LH global head and Tails

        if (i<n){
                if (type[i]==1){
			k2=0;
                        for (int cnti=0; cnti<i;cnti++){
                                if (type[cnti]==1){
                                        k2=k2+1;
                                }
                        }
                        if (nc_lh_flag[k2]==1){
                                for (m=0;m<n_lh_g;m++){
                                        r_lho[0] = r[i*3] + a[i*3]*LH_g_pos[m*3] + b[i*3]*LH_g_pos[m*3+1] + c[i*3]*LH_g_pos[m*3+2];
                                        r_lho[1] = r[i*3+1] + a[i*3+1]*LH_g_pos[m*3] + b[i*3+1]*LH_g_pos[m*3+1] + c[i*3+1]*LH_g_pos[m*3+2];
                                        r_lho[2] = r[i*3+2] + a[i*3+2]*LH_g_pos[m*3] + b[i*3+2]*LH_g_pos[m*3+1] + c[i*3+2]*LH_g_pos[m*3+2];

                                        if (j<n_tail){
                                                for (xi = 0; xi<3; xi++){
                                                        r_t_tmp[xi] = r_t[j*3+xi];
                                                        LH_g_pos_tmp[xi] = LH_g_pos[m*3+xi];
                                                        a_tmp[xi] = a[i*3+xi];
                                                        b_tmp[xi] = b[i*3+xi];
                                                        c_tmp[xi] = c[i*3+xi];
                                                }
                                                Force_Ele_Vdw_cuda(LH_q[m], t_q[j], 1/(4*PI*k_e), debye,k_ex, LH_vdw_ht[m], r_lho, r_t_tmp,force_projection1, force_projection2, Energy_m[i*maxrtlh+j]);

                                                force_m[i*2*maxrtlh*3+j*3] += force_projection1[0];
                                                force_m[i*2*maxrtlh*3+j*3+1] += force_projection1[1];
                                                force_m[i*2*maxrtlh*3+j*3+2] += force_projection1[2];

                                                torque_due_to_force_relative_cuda(force_projection1, LH_g_pos_tmp, a_tmp, b_tmp, c_tmp, torque_hgl);
                                                torque_m[i*2*maxrtlh*3+j*3] += torque_hgl[0];
                                                torque_m[i*2*maxrtlh*3+j*3+1] += torque_hgl[1];
                                                torque_m[i*2*maxrtlh*3+j*3+2] += torque_hgl[2];


                                                if (t_fix[j]==0){
                                                        t_force_m[j*2*maxrtlh*3+maxrtlh*3+i*3] += force_projection2[0];
                                                        t_force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] += force_projection2[1];
                                                        t_force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] += force_projection2[2];
                                                }else{
                                                        k=0;
                                                        for (int ccnt=0; ccnt<n; ccnt++){
                                                                if (k==(t_grp[j]-1)/10 and type[ccnt]==1){
                                                                        index_c = ccnt;
                                                                        break;
                                                                }
                                                                if(type[ccnt]==1){
                                                                        k=k+1;
                                                                }
                                                        }

                                                        for (int ix = 0; ix<3; ix++){
                                                                r_tmp[ix] = r[index_c*3+ix];
                                                                a_tmp[ix] = a[index_c*3+ix];
                                                                b_tmp[ix] = b[index_c*3+ix];
                                                                c_tmp[ix] = c[index_c*3+ix];
                                                        }
                                                        torque_due_to_force_cuda(force_projection2, r_t_tmp, r_tmp, a_tmp, b_tmp, c_tmp, torque2);
							temp_force_m[i*2*maxrtlh*3+j*3] += force_projection2[0];
			                                temp_force_m[i*2*maxrtlh*3+j*3+1] += force_projection2[1];
                        			        temp_force_m[i*2*maxrtlh*3+j*3+2] += force_projection2[2];
			                                temp_force_m[i*2*maxrtlh*3+maxrtlh*3+j*3] += torque2[0];
                        			        temp_force_m[i*2*maxrtlh*3+maxrtlh*3+j*3+1] += torque2[1];
			                                temp_force_m[i*2*maxrtlh*3+maxrtlh*3+j*3+2] += torque2[2];
						}
                                        }
                                }
                        }
                }
        }

	__threadfence();
	//Electrostatic and Excluded Volum between LH global head and Linker DNA

        if (i<n){
                if (type[i]==1){
			k=0;
                        for (int cnti=0; cnti<i;cnti++){
                                if (type[cnti]==1){
                                        k=k+1;
                                }
                        }
			if (nc_lh_flag[k]==1){
				for (l=0;l<n_lh_g;l++){
                                        r_lho[0] = r[i*3] + a[i*3]*LH_g_pos[l*3] + b[i*3]*LH_g_pos[l*3+1] + c[i*3]*LH_g_pos[l*3+2];
                                        r_lho[1] = r[i*3+1] + a[i*3+1]*LH_g_pos[l*3] + b[i*3+1]*LH_g_pos[l*3+1] + c[i*3+1]*LH_g_pos[l*3+2];
                                        r_lho[2] = r[i*3+2] + a[i*3+2]*LH_g_pos[l*3] + b[i*3+2]*LH_g_pos[l*3+1] + c[i*3+2]*LH_g_pos[l*3+2];
                                        if (j<n){
                                                if (type[j]==0){
                                                        for (xi = 0; xi<3; xi++){
                                                                r_tmp[xi] = r[j*3+xi];
                                                                LH_g_pos_tmp[xi] = LH_g_pos[l*3+xi];
                                                                a_tmp[xi] = a[i*3+xi];
                                                                b_tmp[xi] = b[i*3+xi];
                                                                c_tmp[xi] = c[i*3+xi];
                                                        }
                                                        Force_Ele_Vdw_cuda(LH_q[l], q_l, 1/(4*PI*k_e), debye, k_ex, LH_vdw_hl[l], r_lho, r_tmp,force_projection1, force_projection2, Energy_m[i*maxrtlh+j]);

                                                        force_m[i*2*maxrtlh*3+j*3] += force_projection1[0];
                                                        force_m[i*2*maxrtlh*3+j*3+1] += force_projection1[1];
                                                        force_m[i*2*maxrtlh*3+j*3+2] += force_projection1[2];
                                                        force_m[j*2*maxrtlh*3+maxrtlh*3+i*3] += force_projection2[0];
                                                        force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] += force_projection2[1];
                                                        force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] += force_projection2[2];

                                                        torque_due_to_force_relative_cuda(force_projection1, LH_g_pos_tmp, a_tmp, b_tmp, c_tmp, torque_hgl);
                                                        torque_m[i*2*maxrtlh*3+j*3] += torque_hgl[0];
                                                        torque_m[i*2*maxrtlh*3+j*3+1] += torque_hgl[1];
                                                        torque_m[i*2*maxrtlh*3+j*3+2] += torque_hgl[2];

                                                }
                                        }
                                }
                        }
                }
        }

	__threadfence();

	//Electrostatic and Excluded Volum between LH global head and core

        if (i<n){
                if (type[i]==1){
			k=0;
                        for (int cnti=0; cnti<i;cnti++){
                                if (type[cnti]==1){
                                        k=k+1;
                                }
                        }
			if (nc_lh_flag[k]==1){
                                for (l=0;l<n_lh_g;l++){
                                        r_lho[0] = r[i*3] + a[i*3]*LH_g_pos[l*3] + b[i*3]*LH_g_pos[l*3+1] + c[i*3]*LH_g_pos[l*3+2];
                                        r_lho[1] = r[i*3+1] + a[i*3+1]*LH_g_pos[l*3] + b[i*3+1]*LH_g_pos[l*3+1] + c[i*3+1]*LH_g_pos[l*3+2];
                                        r_lho[2] = r[i*3+2] + a[i*3+2]*LH_g_pos[l*3] + b[i*3+2]*LH_g_pos[l*3+1] + c[i*3+2]*LH_g_pos[l*3+2];
                                        if (j<n){
                                                if (i!=j and type[j]==1){
                                                        for (m=0; m<Nq; m++){
                                                                z[0] = r[j*3]+a[j*3]*core_pos[m*3]+b[j*3]*core_pos[m*3+1]+c[j*3]*core_pos[m*3+2];
                                                                z[1] = r[j*3+1]+a[j*3+1]*core_pos[m*3]+b[j*3+1]*core_pos[m*3+1]+c[j*3+1]*core_pos[m*3+2];
                                                                z[2] = r[j*3+2]+a[j*3+2]*core_pos[m*3]+b[j*3+2]*core_pos[m*3+1]+c[j*3+2]*core_pos[m*3+2];

                                                                Force_Ele_Vdw_cuda(LH_q[l], core_q[m], 1/(4*PI*k_e), debye, k_ex, LH_vdw_hc[l], r_lho, z, force_projection1, force_projection2, Energy_m[i*maxrtlh+j]);

                                                                for (xi = 0; xi<3; xi++){
                                                                        LH_g_pos_tmp[xi] = LH_g_pos[l*3+xi];
                                                                        a_tmp[xi] = a[i*3+xi];
                                                                        b_tmp[xi] = b[i*3+xi];
                                                                        c_tmp[xi] = c[i*3+xi];
                                                                }

                                                                force_m[i*2*maxrtlh*3+j*3] += force_projection1[0];
                                                                force_m[i*2*maxrtlh*3+j*3+1] += force_projection1[1];
                                                                force_m[i*2*maxrtlh*3+j*3+2] += force_projection1[2];
                                                                force_m[j*2*maxrtlh*3+maxrtlh*3+i*3] += force_projection2[0];
                                                                force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] += force_projection2[1];
                                                                force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] += force_projection2[2];

                                                                torque_due_to_force_relative_cuda(force_projection1, LH_g_pos_tmp, a_tmp, b_tmp, c_tmp, torque1);
                                                                torque_m[i*2*maxrtlh*3+j*3] += torque1[0];
                                                                torque_m[i*2*maxrtlh*3+j*3+1] += torque1[1];
                                                                torque_m[i*2*maxrtlh*3+j*3+2] += torque1[2];
                                                                for (xi = 0; xi<3; xi++){
                                                                        core_pos_tmp[xi] = core_pos[m*3+xi];
                                                                        a_tmp[xi] = a[j*3+xi];
                                                                        b_tmp[xi] = b[j*3+xi];
                                                                        c_tmp[xi] = c[j*3+xi];
                                                                }

                                                                torque_due_to_force_relative_cuda(force_projection2, core_pos_tmp, a_tmp, b_tmp, c_tmp, torque2);
                                                                torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3] += torque2[0];
                                                                torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] += torque2[1];
                                                                torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] += torque2[2];
                                                        }
                                                }
                                        }
                                }
                        }
                }
        }

	__threadfence();

	//Electrostatic and Excluded Volume between LH global head and LH


        if (i<n){
                if (type[i]==1){
			k2=0;
                        for (int cnti=0; cnti<i;cnti++){
                                if (type[cnti]==1){
                                        k2=k2+1;
                                }
                        }
                        if (nc_lh_flag[k2]==1){
				count = 0;
                                for (int cntc=0;cntc<k2;cntc++){
                                        if (nc_lh_flag[cntc]==1){
                                                count = count+1;
                                        }
                                }
                                for (m=0;m<n_lh_g;m++){
                                        r_lho[0] = r[i*3] + a[i*3]*LH_g_pos[m*3] + b[i*3]*LH_g_pos[m*3+1] + c[i*3]*LH_g_pos[m*3+2];
                                        r_lho[1] = r[i*3+1] + a[i*3+1]*LH_g_pos[m*3] + b[i*3+1]*LH_g_pos[m*3+1] + c[i*3+1]*LH_g_pos[m*3+2];
                                        r_lho[2] = r[i*3+2] + a[i*3+2]*LH_g_pos[m*3] + b[i*3+2]*LH_g_pos[m*3+1] + c[i*3+2]*LH_g_pos[m*3+2];

                                        if (j<n_LH){
                                                if (j/(n_lh_n +n_lh_c) != count){
                                                        index2 = j%(n_lh_n+n_lh_c);
                                                        if (index2 < n_lh_n){
                                                                index2 = index2;
                                                        }else{
                                                                index2 = index2 + n_lh_g;
                                                        }

                                                        for (xi = 0; xi<3; xi++){
                                                                r_lh_tmp2[xi] = r_lh[j*3+xi];
                                                                LH_g_pos_tmp[xi] = LH_g_pos[m*3+xi];
                                                                a_tmp[xi] = a[i*3+xi];
                                                                b_tmp[xi] = b[i*3+xi];
                                                                c_tmp[xi] = c[i*3+xi];
                                                        }

                                                        Force_Ele_Vdw_cuda(LH_q[m], LH_q[index2], 1/(4*PI*k_e), debye,k_ex, (LH_vdw_hh[m]+LH_vdw_hh[index2])/2, r_lho, r_lh_tmp2,force_projection1, force_projection2, Energy_m[i*maxrtlh+j]);

							force_m[i*2*maxrtlh*3+j*3] += force_projection1[0];
                                                        force_m[i*2*maxrtlh*3+j*3+1] += force_projection1[1];
                                                        force_m[i*2*maxrtlh*3+j*3+2] += force_projection1[2];

							LH_force_m[j*2*maxrtlh*3+maxrtlh*3+i*3] += force_projection2[0];
                			                LH_force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] += force_projection2[1];
		                        	        LH_force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] += force_projection2[2];

                                                        torque_due_to_force_relative_cuda(force_projection1, LH_g_pos_tmp, a_tmp, b_tmp, c_tmp, torque_hgl);
                                                        torque_m[i*2*maxrtlh*3+j*3] += torque_hgl[0];
                                                        torque_m[i*2*maxrtlh*3+j*3+1] += torque_hgl[1];
                                                        torque_m[i*2*maxrtlh*3+j*3+2] += torque_hgl[2];

                                                }
                                        }
                                }
                                
                        }
                }
        }

	__threadfence();

	//Electrostatic and Excluded Volume between LH global head and LH global head

        if (i<n){
                if (type[i]==1){
			k2=0;
                        for (int cnti=0; cnti<i;cnti++){
                                if (type[cnti]==1){
                                        k2=k2+1;
                                }
                        }
                        if (nc_lh_flag[k2]==1){
                                for (m=0;m<n_lh_g;m++){
                                        r_lho[0] = r[i*3] + a[i*3]*LH_g_pos[m*3] + b[i*3]*LH_g_pos[m*3+1] + c[i*3]*LH_g_pos[m*3+2];
                                        r_lho[1] = r[i*3+1] + a[i*3+1]*LH_g_pos[m*3] + b[i*3+1]*LH_g_pos[m*3+1] + c[i*3+1]*LH_g_pos[m*3+2];
                                        r_lho[2] = r[i*3+2] + a[i*3+2]*LH_g_pos[m*3] + b[i*3+2]*LH_g_pos[m*3+1] + c[i*3+2]*LH_g_pos[m*3+2];

                                        k = k2+1;
                                        if (j>=i+1 and j<n){
                                                if (type[j]==1){
                                                        if (nc_lh_flag[k]==1){
                                                                for (m1=0;m1<n_lh_g;m1++){
                                                                        r_lho2[0] = r[j*3] + a[j*3]*LH_g_pos[m1*3] + b[j*3]*LH_g_pos[m1*3+1] + c[j*3]*LH_g_pos[m1*3+2];
                                                                        r_lho2[1] = r[j*3+1] + a[j*3+1]*LH_g_pos[m1*3] + b[j*3+1]*LH_g_pos[m1*3+1] + c[j*3+1]*LH_g_pos[m1*3+2];
                                                                        r_lho2[2] = r[j*3+2] + a[j*3+2]*LH_g_pos[m1*3] + b[j*3+2]*LH_g_pos[m1*3+1] + c[j*3+2]*LH_g_pos[m1*3+2];

                                                                        Force_Ele_Vdw_cuda(LH_q[m], LH_q[m1], 1/(4*PI*k_e), debye,k_ex, (LH_vdw_hh[m]+LH_vdw_hh[m1])/2, r_lho, r_lho2,force_projection1, force_projection2, Energy_m[i*maxrtlh+j]);

									force_m[i*2*maxrtlh*3+j*3] += force_projection1[0];
                		                                        force_m[i*2*maxrtlh*3+j*3+1] += force_projection1[1];
		                                                        force_m[i*2*maxrtlh*3+j*3+2] += force_projection1[2];

									force_m[j*2*maxrtlh*3+maxrtlh*3+i*3] += force_projection2[0];
					                                force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] += force_projection2[1];
					                                force_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] += force_projection2[2];

                                                                        for (xi = 0; xi<3; xi++){
                                                                                LH_g_pos_tmp[xi] = LH_g_pos[m*3+xi];
                                                                                a_tmp[xi] = a[i*3+xi];
                                                                                b_tmp[xi] = b[i*3+xi];
                                                                                c_tmp[xi] = c[i*3+xi];
                                                                        }

                                                                        torque_due_to_force_relative_cuda(force_projection1, LH_g_pos_tmp, a_tmp, b_tmp, c_tmp, torque_hgl);
                                                                        torque_m[i*2*maxrtlh*3+j*3] += torque_hgl[0];
                                                                        torque_m[i*2*maxrtlh*3+j*3+1] += torque_hgl[1];
                                                                        torque_m[i*2*maxrtlh*3+j*3+2] += torque_hgl[2];

                                                                        for (xi = 0; xi<3; xi++){
                                                                                LH_g_pos_tmp[xi] = LH_g_pos[m1*3+xi];
                                                                                a_tmp[xi] = a[j*3+xi];
                                                                                b_tmp[xi] = b[j*3+xi];
                                                                                c_tmp[xi] = c[j*3+xi];
                                                                        }

                                                                        torque_due_to_force_relative_cuda(force_projection2, LH_g_pos_tmp, a_tmp, b_tmp, c_tmp, torque_hgl);
                                                                        torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3] += torque_hgl[0];
                                                                        torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3+1] += torque_hgl[1];
                                                                        torque_m[j*2*maxrtlh*3+maxrtlh*3+i*3+2] += torque_hgl[2];

                                                                }
                                                        }
                                                }
                                        }
                                }

                        }
                        
                }
        }

	__threadfence();

	//Stretching between LH global head with C-term


        if (i < n){
                if (type[i]==1){
			k2=0;
                        for (int cnti=0; cnti<i;cnti++){
                                if (type[cnti]==1){
                                        k2=k2+1;
                                }
                        }
                        if (nc_lh_flag[k2]==1){
				k = 0;
                                for (int cntc=0;cntc<k2;cntc++){
                                        if (nc_lh_flag[cntc]==1){
                                                k = k+1;
                                        }
                                }
                                if (j < n_lh_g){
                                        if (LH_conn[j+n_lh_n]==1){
                                                r_lho[0] = r[i*3] + a[i*3]*LH_g_pos[j*3] + b[i*3]*LH_g_pos[j*3+1] + c[i*3]*LH_g_pos[j*3+2];
                                                r_lho[1] = r[i*3+1] + a[i*3+1]*LH_g_pos[j*3] + b[i*3+1]*LH_g_pos[j*3+1] + c[i*3+1]*LH_g_pos[j*3+2];
                                                r_lho[2] = r[i*3+2] + a[i*3+2]*LH_g_pos[j*3] + b[i*3+2]*LH_g_pos[j*3+1] + c[i*3+2]*LH_g_pos[j*3+2];

                                                distance = (r_lh[k*nlh*3]-r_lho[0])*(r_lh[k*nlh*3]-r_lho[0])+(r_lh[k*nlh*3+1]-r_lho[1])*(r_lh[k*nlh*3+1]-r_lho[1])+(r_lh[k*nlh*3+2]-r_lho[2])*(r_lh[k*nlh*3+2]-r_lho[2]);
						distance = sqrt(distance);

                                                Energy_m[i*maxrtlh+j] = Energy_m[i*maxrtlh+j] + LH_kstr[j+n_lh_n]*(distance-LH_streq[j+n_lh_n])*(distance-LH_streq[j+n_lh_n])/2;


                                                Stri[0] = (distance - LH_streq[j+n_lh_n])*(r_lh[k*nlh*3]-r_lho[0])/distance;
                                                Stri[1] = (distance - LH_streq[j+n_lh_n])*(r_lh[k*nlh*3+1]-r_lho[1])/distance;
                                                Stri[2] = (distance - LH_streq[j+n_lh_n])*(r_lh[k*nlh*3+2]-r_lho[2])/distance;


                                                LH_force_m[k*nlh*2*maxrtlh*3+j*3] -= LH_kstr[j+n_lh_n]*Stri[0]*10;
                                                LH_force_m[k*nlh*2*maxrtlh*3+j*3+1] -= LH_kstr[j+n_lh_n]*Stri[1]*10;
                                                LH_force_m[k*nlh*2*maxrtlh*3+j*3+2] -= LH_kstr[j+n_lh_n]*Stri[2]*10;

                                                force_m[i*2*maxrtlh*3+j*3] += LH_kstr[j+n_lh_n]*Stri[0]*10;
                                                force_m[i*2*maxrtlh*3+j*3+1] += LH_kstr[j+n_lh_n]*Stri[1]*10;
                                                force_m[i*2*maxrtlh*3+j*3+2] += LH_kstr[j+n_lh_n]*Stri[2]*10;

                                                fa = LH_kstr[j+n_lh_n]*(a[i*3]*Stri[0]+a[i*3+1]*Stri[1]+a[i*3+2]*Stri[2]);
                                                fb = LH_kstr[j+n_lh_n]*(b[i*3]*Stri[0]+b[i*3+1]*Stri[1]+b[i*3+2]*Stri[2]);
                                                fc = LH_kstr[j+n_lh_n]*(c[i*3]*Stri[0]+c[i*3+1]*Stri[1]+c[i*3+2]*Stri[2]);
                                                torque_m[i*2*maxrtlh*3+j*3] += fc*LH_g_pos[j*3+1] - fb*LH_g_pos[j*3+2];
                                                torque_m[i*2*maxrtlh*3+j*3+1] += fa*LH_g_pos[j*3+2] - fc*LH_g_pos[j*3];
                                                torque_m[i*2*maxrtlh*3+j*3+2] += fb*LH_g_pos[j*3] - fa*LH_g_pos[j*3+1];

                                        }
                                }
                        }
                }
        }

}

__global__ void force_torque_tmp(int n3, int n_tail3, int n_LH3, double* force, double* torque, double* t_force, double* LH_force, double* force_n, double* torque_n, double* t_force_n, double* LH_force_n, double* force_tmp, double* torque_tmp, double* t_force_tmp, double* LH_force_tmp){

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n3){
		force_tmp[i] = 0.5*(force[i]+force_n[i]);
		torque_tmp[i] = 0.5*(torque[i]+torque_n[i]);
	}
	if (i<n_tail3){
		t_force_tmp[i] = 0.5*(t_force[i]+t_force_n[i]);
	}
	if (i<n_LH3){
		LH_force_tmp[i] = 0.5*(LH_force[i]+LH_force_n[i]);
	}

}

__global__ void final_updates(int n3, int n_tail3, int n_LH3, double* r, double* r_n, double* a, double* a_n, double* b, double* b_n, double* c, double* c_n, double* r_t, double* r_t_n, double* r_lh, double* r_lh_n){

	int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < n3){
		r[i] = r_n[i];
		a[i] = a_n[i];
		b[i] = b_n[i];
		c[i] = c_n[i];
	}

	if(i < n_tail3){
		r_t[i] = r_t_n[i];
	}

	if(i < n_LH3){
		r_lh[i] = r_lh_n[i];
	}

}


extern "C++" void cuda_application_init_D_Chol(int n3){


	size_t bytes_D = n3*n3*sizeof(double);
        size_t bytes_Chol = n3*n3*sizeof(double);

	cudaMalloc(&d_D, bytes_D);
        cudaMalloc(&d_Chol, bytes_Chol);

}

extern "C++" void cuda_application_init_data(int n_c, int nc3, int n, int n3, int* type, double* r, double* a, double* b, double* c, double* alpha, double* beta, double* gamma, double* length, double* a_dna, double* b_dna, double* c_dna, double* alpha_p, double* beta_p, double* gamma_p, double h, double g, double s, double* phi_o, double debyell, double debye, double q_l, double k_e, double k_ex, double k_h1, double sigma_DNA_DNA, double sigma_DNA_Core, double sigma_Core_Core, double sigma_Tail_Tail, double sigma_Tail_Linker, double sigma_Tail_Core, int Nq, int Nq3, double* core_pos, double* core_q, int n_t, int n_tail, int n_tail3, double* tail_pos, int* tail_fix, int* nc_t_flag, double* r_t, double* beta_t, double* h_t, double* g_t, double* lo_t, double* beta_o_t, double* t_q, double* t_rad, int* t_grp, int* t_fix, int n_lh_n, int n_lh_g, int n_lh_c, int n_LH, int n_LH3, double* LH_g_pos, int* LH_conn, int* nc_lh_flag, double* beta_lh, double* r_lh, double* LH_q, double* LH_vdw_hh, double* LH_vdw_hc, double* LH_vdw_hl, double* LH_vdw_ht, double* LH_kstr, double* LH_kben, double* LH_streq, double* LH_betaeq, double* LH_force, double* t_force, double* force, double* torque, double* Energy, double* r_all, double* rad_all){


	int n_D = n+n_tail+n_LH;
	int n_D3 = n_D*3;

	size_t bytes = sizeof(double);
        size_t bytes_ni = n*sizeof(int);
        size_t bytes_nd = n*sizeof(double);
        size_t bytes_n3 = n3*sizeof(double);
        size_t bytes_nc = n_c*sizeof(double);
        size_t bytes_nc3 = nc3*sizeof(double);
        size_t bytes_Nq = Nq*sizeof(double);
        size_t bytes_Nq3 = Nq3*sizeof(double);

        size_t bytes_t = n_t*sizeof(int);
        size_t bytes_t3 = n_t*3*sizeof(double);
        size_t bytes_tail = n_tail*sizeof(double);
        size_t bytes_tail3 = n_tail3*sizeof(double);
        size_t bytes_tail_int = n_tail*sizeof(int);
        size_t bytes_nc_int = n_c*sizeof(int);

        size_t bytes_lh_g = n_lh_g*3*sizeof(double);
        size_t bytes_lh_conn = (n_lh_g+n_lh_c+n_lh_n)*sizeof(int);
        size_t bytes_LH = n_LH*sizeof(double);
        size_t bytes_LH3 = n_LH3*sizeof(double);
        size_t bytes_lh_d = (n_lh_g+n_lh_c+n_lh_n)*sizeof(double);

	size_t bytes_n_D = n_D*sizeof(double);
	size_t bytes_n_D3 = n_D3*sizeof(double);

        int maxrtlh;

        if (n>n_LH){
                if (n>n_tail){
                        maxrtlh=n;
                }else{
                        maxrtlh=n_tail;
                }
        }else{
                if (n_LH>n_tail){
                        maxrtlh=n_LH;
                }else{
                        maxrtlh=n_tail;
                }
        }

        size_t bytes_r_t_lh_m = maxrtlh*maxrtlh*6*sizeof(double);

	size_t bytes_r_t_lh_m_E = maxrtlh*maxrtlh*sizeof(double);

	//Allocate memory on device side

        cudaMalloc(&d_Energy, bytes);
        cudaMalloc(&d_type, bytes_ni);
        cudaMalloc(&d_r, bytes_n3);
        cudaMalloc(&d_a, bytes_n3);
        cudaMalloc(&d_b, bytes_n3);
        cudaMalloc(&d_c, bytes_n3);
        cudaMalloc(&d_alpha, bytes_nd);
        cudaMalloc(&d_beta, bytes_nd);
        cudaMalloc(&d_gamma, bytes_nd);
        cudaMalloc(&d_length, bytes_nd);
        cudaMalloc(&d_a_dna, bytes_nc3);
        cudaMalloc(&d_b_dna, bytes_nc3);
        cudaMalloc(&d_c_dna, bytes_nc3);
        cudaMalloc(&d_alpha_p, bytes_nc);
        cudaMalloc(&d_beta_p, bytes_nc);
        cudaMalloc(&d_gamma_p, bytes_nc);
        cudaMalloc(&d_phi_o, bytes_nd);
        cudaMalloc(&d_core_pos, bytes_Nq3);
        cudaMalloc(&d_core_q, bytes_Nq);
        cudaMalloc(&d_force, bytes_n3);
        cudaMalloc(&d_torque, bytes_n3);
	cudaMalloc(&d_force_n, bytes_n3);
        cudaMalloc(&d_torque_n, bytes_n3);

	cudaMalloc(&d_force_tmp, bytes_n3);
        cudaMalloc(&d_torque_tmp, bytes_n3);

	cudaMalloc(&d_r_all, bytes_n_D3);
	cudaMalloc(&d_rad_all, bytes_n_D);
	cudaMalloc(&d_d_theta, bytes_n3);
	cudaMalloc(&d_rd, bytes_n_D3);

	cudaMalloc(&d_r_n, bytes_n3);
        cudaMalloc(&d_a_n, bytes_n3);
        cudaMalloc(&d_b_n, bytes_n3);
        cudaMalloc(&d_c_n, bytes_n3);
        cudaMalloc(&d_alpha_n, bytes_nd);
        cudaMalloc(&d_beta_n, bytes_nd);
        cudaMalloc(&d_gamma_n, bytes_nd);
        cudaMalloc(&d_length_n, bytes_nd);
        cudaMalloc(&d_a_dna_n, bytes_nc3);
        cudaMalloc(&d_b_dna_n, bytes_nc3);
        cudaMalloc(&d_c_dna_n, bytes_nc3);
        cudaMalloc(&d_alpha_p_n, bytes_nc);
        cudaMalloc(&d_beta_p_n, bytes_nc);
        cudaMalloc(&d_gamma_p_n, bytes_nc);

        cudaMalloc(&d_tail_pos, bytes_t3);
        cudaMalloc(&d_tail_fix, bytes_t);
        cudaMalloc(&d_nc_t_flag, bytes_nc_int);
        cudaMalloc(&d_r_t, bytes_tail3);
        cudaMalloc(&d_beta_t, bytes_tail);
        cudaMalloc(&d_h_t, bytes_tail);
        cudaMalloc(&d_g_t, bytes_tail);
        cudaMalloc(&d_lo_t, bytes_tail);
        cudaMalloc(&d_beta_o_t, bytes_tail);
        cudaMalloc(&d_t_q, bytes_tail);
        cudaMalloc(&d_t_rad, bytes_tail);
        cudaMalloc(&d_t_grp, bytes_tail_int);
        cudaMalloc(&d_t_fix, bytes_tail_int);
        cudaMalloc(&d_t_force, bytes_tail3);
	cudaMalloc(&d_t_force_n, bytes_tail3);

	cudaMalloc(&d_t_force_tmp, bytes_tail3);

	cudaMalloc(&d_r_t_n, bytes_tail3);
        cudaMalloc(&d_beta_t_n, bytes_tail);

        cudaMalloc(&d_LH_g_pos, bytes_lh_g);
        cudaMalloc(&d_LH_conn, bytes_lh_conn);
        cudaMalloc(&d_nc_lh_flag, bytes_nc_int);
        cudaMalloc(&d_beta_lh, bytes_LH);
        cudaMalloc(&d_r_lh, bytes_LH3);
        cudaMalloc(&d_LH_q, bytes_lh_d);
        cudaMalloc(&d_LH_vdw_hh, bytes_lh_d);
        cudaMalloc(&d_LH_vdw_hc, bytes_lh_d);
        cudaMalloc(&d_LH_vdw_hl, bytes_lh_d);
        cudaMalloc(&d_LH_vdw_ht, bytes_lh_d);
        cudaMalloc(&d_LH_kstr, bytes_lh_d);
        cudaMalloc(&d_LH_kben, bytes_lh_d);
        cudaMalloc(&d_LH_streq, bytes_lh_d);
        cudaMalloc(&d_LH_betaeq, bytes_lh_d);
        cudaMalloc(&d_LH_force, bytes_LH3);
	cudaMalloc(&d_LH_force_n, bytes_LH3);

	cudaMalloc(&d_LH_force_tmp, bytes_LH3);

	cudaMalloc(&d_beta_lh_n, bytes_LH);
        cudaMalloc(&d_r_lh_n, bytes_LH3);

        cudaMalloc(&d_force_m, bytes_r_t_lh_m);
        cudaMalloc(&d_torque_m, bytes_r_t_lh_m);
        cudaMalloc(&d_t_force_m, bytes_r_t_lh_m);
        cudaMalloc(&d_LH_force_m, bytes_r_t_lh_m);
        cudaMalloc(&d_temp_force_m, bytes_r_t_lh_m);

	cudaMalloc(&d_Energy_m, bytes_r_t_lh_m_E);

	//Copy data to the device

        cudaMemcpy(d_Energy, Energy, bytes,cudaMemcpyHostToDevice);
        cudaMemcpy(d_type, type, bytes_ni, cudaMemcpyHostToDevice);
        cudaMemcpy(d_r, r, bytes_n3, cudaMemcpyHostToDevice);
        cudaMemcpy(d_a, a, bytes_n3, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, bytes_n3, cudaMemcpyHostToDevice);
        cudaMemcpy(d_c, c, bytes_n3, cudaMemcpyHostToDevice);
        cudaMemcpy(d_alpha, alpha, bytes_nd, cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta, beta, bytes_nd, cudaMemcpyHostToDevice);
        cudaMemcpy(d_gamma, gamma, bytes_nd, cudaMemcpyHostToDevice);
        cudaMemcpy(d_length, length, bytes_nd, cudaMemcpyHostToDevice);
        cudaMemcpy(d_a_dna, a_dna, bytes_nc3, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b_dna, b_dna, bytes_nc3, cudaMemcpyHostToDevice);
        cudaMemcpy(d_c_dna, c_dna, bytes_nc3, cudaMemcpyHostToDevice);
        cudaMemcpy(d_alpha_p, alpha_p, bytes_nc, cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta_p, beta_p, bytes_nc, cudaMemcpyHostToDevice);
        cudaMemcpy(d_gamma_p, gamma_p, bytes_nc, cudaMemcpyHostToDevice);
        cudaMemcpy(d_phi_o, phi_o, bytes_nd, cudaMemcpyHostToDevice);
        cudaMemcpy(d_core_pos, core_pos, bytes_Nq3, cudaMemcpyHostToDevice);
        cudaMemcpy(d_core_q, core_q, bytes_Nq, cudaMemcpyHostToDevice);
        cudaMemcpy(d_force, force, bytes_n3, cudaMemcpyHostToDevice);
        cudaMemcpy(d_torque, torque, bytes_n3, cudaMemcpyHostToDevice);

	cudaMemcpy(d_r_all, r_all, bytes_n_D3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rad_all, rad_all, bytes_n_D, cudaMemcpyHostToDevice);

        cudaMemcpy(d_tail_pos, tail_pos, bytes_t3, cudaMemcpyHostToDevice);
        cudaMemcpy(d_tail_fix, tail_fix, bytes_t, cudaMemcpyHostToDevice);
        cudaMemcpy(d_nc_t_flag, nc_t_flag, bytes_nc_int, cudaMemcpyHostToDevice);
        cudaMemcpy(d_r_t, r_t, bytes_tail3, cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta_t, beta_t, bytes_tail, cudaMemcpyHostToDevice);
        cudaMemcpy(d_h_t, h_t, bytes_tail, cudaMemcpyHostToDevice);
        cudaMemcpy(d_g_t, g_t, bytes_tail, cudaMemcpyHostToDevice);
        cudaMemcpy(d_lo_t, lo_t, bytes_tail, cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta_o_t, beta_o_t, bytes_tail, cudaMemcpyHostToDevice);
        cudaMemcpy(d_t_q, t_q, bytes_tail, cudaMemcpyHostToDevice);
        cudaMemcpy(d_t_rad, t_rad, bytes_tail, cudaMemcpyHostToDevice);
        cudaMemcpy(d_t_grp, t_grp, bytes_tail_int, cudaMemcpyHostToDevice);
        cudaMemcpy(d_t_fix, t_fix, bytes_tail_int, cudaMemcpyHostToDevice);
        cudaMemcpy(d_t_force, t_force, bytes_tail3, cudaMemcpyHostToDevice);

        cudaMemcpy(d_LH_g_pos, LH_g_pos, bytes_lh_g, cudaMemcpyHostToDevice);
        cudaMemcpy(d_LH_conn, LH_conn, bytes_lh_conn, cudaMemcpyHostToDevice);
        cudaMemcpy(d_nc_lh_flag, nc_lh_flag, bytes_nc_int, cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta_lh, beta_lh, bytes_LH, cudaMemcpyHostToDevice);
        cudaMemcpy(d_r_lh, r_lh, bytes_LH3, cudaMemcpyHostToDevice);
        cudaMemcpy(d_LH_q, LH_q, bytes_lh_d, cudaMemcpyHostToDevice);
        cudaMemcpy(d_LH_vdw_hh, LH_vdw_hh, bytes_lh_d, cudaMemcpyHostToDevice);
        cudaMemcpy(d_LH_vdw_hc, LH_vdw_hc, bytes_lh_d, cudaMemcpyHostToDevice);
        cudaMemcpy(d_LH_vdw_hl, LH_vdw_hl, bytes_lh_d, cudaMemcpyHostToDevice);
        cudaMemcpy(d_LH_vdw_ht, LH_vdw_ht, bytes_lh_d, cudaMemcpyHostToDevice);
        cudaMemcpy(d_LH_kstr, LH_kstr, bytes_lh_d, cudaMemcpyHostToDevice);
        cudaMemcpy(d_LH_kben, LH_kben, bytes_lh_d, cudaMemcpyHostToDevice);
        cudaMemcpy(d_LH_streq, LH_streq, bytes_lh_d, cudaMemcpyHostToDevice);
        cudaMemcpy(d_LH_betaeq, LH_betaeq, bytes_lh_d, cudaMemcpyHostToDevice);
        cudaMemcpy(d_LH_force, LH_force, bytes_LH3, cudaMemcpyHostToDevice);



}
/*
extern "C++" void cuda_application_D_Chol(int n, int n3, double* r, double a1, double a2, double* rad){


	//Free D and Chol
	cudaFree(d_D);
	cudaFree(d_Chol);

	//initialize cuSolver and cublas
	cusolverDnHandle_t solver_handle;
        cusolverDnCreate(&solver_handle);

        int work_size = 0;

        int *devInfo;
        cudaMalloc(&devInfo, sizeof(int));

	//Device Matrices

        double* d_r;
        double* d_rad;

        size_t bytes = n3*sizeof(double);
        size_t bytes_D = n3*n3*sizeof(double);
        size_t bytes_Chol = n3*n3*sizeof(double);

	//Allocate memory on device side

        cudaMalloc(&d_r, bytes);
        cudaMalloc(&d_rad, bytes);

        cudaMalloc(&d_D, bytes_D);
        cudaMalloc(&d_Chol, bytes_Chol);

	//Copy data to the device

        cudaMemcpy(d_r, r, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_rad, rad, bytes, cudaMemcpyHostToDevice);

	//assign number of blocks and threads

	int threads_per_block = 32;
        dim3 block_size(threads_per_block, threads_per_block);
        dim3 grid_size(n3 / block_size.x + 1 , n3 / block_size.y + 1 );

	//Calculate D

	Diffusion_Tensor_CUDA <<<grid_size, block_size>>> (n, n3, d_r, a1, a2, d_rad, d_D, d_Chol);

	//Calculate Chol

	cusolverDnDpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_UPPER, n3, d_Chol, n3, &work_size);

	double *work;
        cudaMalloc(&work, work_size * sizeof(double));

        cusolverDnDpotrf(solver_handle, CUBLAS_FILL_MODE_UPPER, n3, d_Chol, n3, work, work_size, devInfo);

        Cholesky_Decomposition_mod <<<grid_size, block_size>>> (n3, d_Chol);

	cudaFree(d_r);
        cudaFree(d_rad);

	cudaFree(devInfo);
        cudaFree(work);

	cusolverDnDestroy(solver_handle);

}
*/
/*
extern "C++" void cuda_application_rd(int n, int n3, double* r, double a1, double a2, double* rad, double s2dt, double* p, double* rd){

	//Device Matrices

	double* d_r;
	double* d_rad; 
	double* d_rd;
	double* d_p;

	size_t bytes = n3*sizeof(double);

	//Allocate memory on device side

	cudaMalloc(&d_r, bytes);
	cudaMalloc(&d_rad, bytes);
	cudaMalloc(&d_rd, bytes);
	cudaMalloc(&d_p, bytes);

	//Copy data to the device

	cudaMemcpy(d_r, r, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rad, rad, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_p, p, bytes, cudaMemcpyHostToDevice);

	//assign number of blocks and threads

	int num_thread = 32;
	int num_block = (n3 + num_thread -1) / num_thread;

	//calculate rd

	rd_cal <<<num_block, num_thread>>> (n3, d_rd, d_Chol, d_p, s2dt);

	//copy rd back to the host

	cudaMemcpy(rd, d_rd, bytes, cudaMemcpyDeviceToHost);

	cudaFree(d_r);
	cudaFree(d_rad);
	cudaFree(d_rd);
	cudaFree(d_p);

}
*/
/*
extern "C++" void cuda_application_translate(int n, int n_D3, int n3, double* r_all, double* r, double a1, double a2, double* rad, double del, double* force_global, double* t_force_global, double* LH_force_global, int n_tail3, int n_LH3, double* rd, double* r_t, double* r_lh, double* r_n, double* r_t_n, double* r_lh_n){

        //Device Matrices

        double* d_rall;
        double* d_rad;

	double* d_r;
	double* d_r_n;
	double* d_rd;
	double* d_r_t;
	double* d_r_t_n;
	double* d_r_lh;
	double* d_r_lh_n;
	double* d_force_global;
	double* d_t_force_global;
	double* d_LH_force_global;


        size_t bytes = n_D3*sizeof(double);
	size_t bytes_n = n3*sizeof(double);
	size_t bytes_t = n_tail3*sizeof(double);
	size_t bytes_LH = n_LH3*sizeof(double);



        //Allocate memory on device side

        cudaMalloc(&d_rall, bytes);
        cudaMalloc(&d_rad, bytes);

	cudaMalloc(&d_r, bytes_n);
	cudaMalloc(&d_r_n, bytes_n);
	cudaMalloc(&d_rd, bytes);
	cudaMalloc(&d_r_t, bytes_t);
	cudaMalloc(&d_r_t_n, bytes_t);
	cudaMalloc(&d_r_lh, bytes_LH);
	cudaMalloc(&d_r_lh_n, bytes_LH);
	cudaMalloc(&d_force_global, bytes_n);
	cudaMalloc(&d_t_force_global, bytes_t);
	cudaMalloc(&d_LH_force_global, bytes_LH);


        //Copy data to the device

        cudaMemcpy(d_rall, r_all, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_rad, rad, bytes, cudaMemcpyHostToDevice);

	cudaMemcpy(d_r, r, bytes_n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rd, rd, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_r_t, r_t, bytes_t, cudaMemcpyHostToDevice);
	cudaMemcpy(d_r_lh, r_lh, bytes_LH, cudaMemcpyHostToDevice);
	cudaMemcpy(d_force_global, force_global, bytes_n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_t_force_global, t_force_global, bytes_t, cudaMemcpyHostToDevice);
	cudaMemcpy(d_LH_force_global, LH_force_global, bytes_LH, cudaMemcpyHostToDevice);

	int num_thread = 32;
        int num_block = (n_D3 + num_thread -1) / num_thread;

	translation_cal <<<num_block, num_thread>>> (n_D3, n3, n_tail3, n_LH3, d_r, d_r_n, d_rd, d_r_t, d_r_t_n, d_r_lh, d_r_lh_n, del, d_force_global, d_t_force_global, d_LH_force_global, d_D);

	cudaMemcpy(r_n, d_r_n, bytes_n, cudaMemcpyDeviceToHost);
	cudaMemcpy(r_t_n, d_r_t_n, bytes_t, cudaMemcpyDeviceToHost);
	cudaMemcpy(r_lh_n, d_r_lh_n, bytes_LH, cudaMemcpyDeviceToHost);


        cudaFree(d_rall);
        cudaFree(d_rad);

	cudaFree(d_r);
        cudaFree(d_r_n);
        cudaFree(d_rd);
	cudaFree(d_r_t);
        cudaFree(d_r_t_n);
        cudaFree(d_r_lh);
	cudaFree(d_r_lh_n);
        cudaFree(d_force_global);
        cudaFree(d_t_force_global);
	cudaFree(d_LH_force_global);

}
*/
/*
extern "C++" void force_and_torque_cuda_application(int n_c, int nc3, int n, int n3, int* type, double* r, double* a, double* b, double* c, double* alpha, double* beta, double* gamma, double* length, double* a_dna, double* b_dna, double* c_dna, double* alpha_p, double* beta_p, double* gamma_p, double h, double g, double s, double* phi_o, double debyell, double debye, double q_l, double k_e, double k_ex, double k_h1, double sigma_DNA_DNA, double sigma_DNA_Core, double sigma_Core_Core, double sigma_Tail_Tail, double sigma_Tail_Linker, double sigma_Tail_Core, int Nq, int Nq3, double* core_pos, double* core_q, int n_t, int n_tail, int n_tail3, double* tail_pos, int* tail_fix, int* nc_t_flag, double* r_t, double* beta_t, double* h_t, double* g_t, double* lo_t, double* beta_o_t, double* t_q, double* t_rad, int* t_grp, int* t_fix, int n_lh_n, int n_lh_g, int n_lh_c, int n_LH, int n_LH3, double* LH_g_pos, int* LH_conn, int* nc_lh_flag, double* beta_lh, double* r_lh, double* LH_q, double* LH_vdw_hh, double* LH_vdw_hc, double* LH_vdw_hl, double* LH_vdw_ht, double* LH_kstr, double* LH_kben, double* LH_streq, double* LH_betaeq, double* LH_force, double* t_force, double* force, double* torque, double* Energy){


	//Device Matrices
	int* d_type;
	double* d_r;
	double* d_a;
	double* d_b;
	double* d_c;
	double* d_alpha;
	double* d_beta;
	double* d_gamma;
	double* d_length;
	double* d_a_dna;
	double* d_b_dna;
	double* d_c_dna;
	double* d_alpha_p;
	double* d_beta_p;
	double* d_gamma_p;
	double* d_phi_o;
	double* d_force;
	double* d_torque;
	double* d_Energy;
	double* d_core_pos;
	double* d_core_q;

	double* d_tail_pos;
	int* d_tail_fix;
	int* d_nc_t_flag;
	double* d_r_t;
	double* d_beta_t;
	double* d_h_t;
	double* d_g_t;
	double* d_lo_t;
	double* d_beta_o_t;
	double* d_t_q;
	double* d_t_rad;
	int* d_t_grp;
	int* d_t_fix;
	double* d_t_force;

	double* d_LH_g_pos; 
	int* d_LH_conn; 
	int* d_nc_lh_flag; 
	double* d_beta_lh; 
	double* d_r_lh; 
	double* d_LH_q; 
	double* d_LH_vdw_hh; 
	double* d_LH_vdw_hc; 
	double* d_LH_vdw_hl; 
	double* d_LH_vdw_ht; 
	double* d_LH_kstr; 
	double* d_LH_kben; 
	double* d_LH_streq; 
	double* d_LH_betaeq; 
	double* d_LH_force; 

	double* d_force_m;
	double* d_torque_m;
	double* d_t_force_m;
	double* d_LH_force_m;

	double* d_temp_force_m;

	size_t bytes = sizeof(double);
	size_t bytes_ni = n*sizeof(int);
	size_t bytes_nd = n*sizeof(double);
	size_t bytes_n3 = n3*sizeof(double);
	size_t bytes_nc = n_c*sizeof(double);
	size_t bytes_nc3 = nc3*sizeof(double);
	size_t bytes_Nq = Nq*sizeof(double);
	size_t bytes_Nq3 = Nq3*sizeof(double);
	
	size_t bytes_t = n_t*sizeof(int);
	size_t bytes_t3 = n_t*3*sizeof(double);
	size_t bytes_tail = n_tail*sizeof(double);
	size_t bytes_tail3 = n_tail3*sizeof(double);
	size_t bytes_tail_int = n_tail*sizeof(int);
	size_t bytes_nc_int = n_c*sizeof(int);

	size_t bytes_lh_g = n_lh_g*3*sizeof(double);
	size_t bytes_lh_conn = (n_lh_g+n_lh_c+n_lh_n)*sizeof(int);
	size_t bytes_LH = n_LH*sizeof(double);
	size_t bytes_LH3 = n_LH3*sizeof(double);
	size_t bytes_lh_d = (n_lh_g+n_lh_c+n_lh_n)*sizeof(double);
	
	int maxrtlh;

	if (n>n_LH){
		if (n>n_tail){
			maxrtlh=n;
		}else{
			maxrtlh=n_tail;
		}
	}else{
		if (n_LH>n_tail){
			maxrtlh=n_LH;
		}else{
			maxrtlh=n_tail;
		}
	}

	size_t bytes_r_t_lh_m = maxrtlh*maxrtlh*6*sizeof(double);

	//Allocate memory on device side

	cudaMalloc(&d_Energy, bytes);
        cudaMalloc(&d_type, bytes_ni);
	cudaMalloc(&d_r, bytes_n3);
	cudaMalloc(&d_a, bytes_n3);
	cudaMalloc(&d_b, bytes_n3);
	cudaMalloc(&d_c, bytes_n3);
	cudaMalloc(&d_alpha, bytes_nd);
	cudaMalloc(&d_beta, bytes_nd);
	cudaMalloc(&d_gamma, bytes_nd);
	cudaMalloc(&d_length, bytes_nd);
	cudaMalloc(&d_a_dna, bytes_nc3);
	cudaMalloc(&d_b_dna, bytes_nc3);
	cudaMalloc(&d_c_dna, bytes_nc3);
	cudaMalloc(&d_alpha_p, bytes_nc);
	cudaMalloc(&d_beta_p, bytes_nc);
	cudaMalloc(&d_gamma_p, bytes_nc);
	cudaMalloc(&d_phi_o, bytes_nd);
	cudaMalloc(&d_core_pos, bytes_Nq3);
	cudaMalloc(&d_core_q, bytes_Nq);
	cudaMalloc(&d_force, bytes_n3);
	cudaMalloc(&d_torque, bytes_n3);

	cudaMalloc(&d_tail_pos, bytes_t3);
	cudaMalloc(&d_tail_fix, bytes_t);
	cudaMalloc(&d_nc_t_flag, bytes_nc_int);
	cudaMalloc(&d_r_t, bytes_tail3);
	cudaMalloc(&d_beta_t, bytes_tail);
	cudaMalloc(&d_h_t, bytes_tail);
	cudaMalloc(&d_g_t, bytes_tail);
	cudaMalloc(&d_lo_t, bytes_tail);
	cudaMalloc(&d_beta_o_t, bytes_tail);
	cudaMalloc(&d_t_q, bytes_tail);
	cudaMalloc(&d_t_rad, bytes_tail);
	cudaMalloc(&d_t_grp, bytes_tail_int);
	cudaMalloc(&d_t_fix, bytes_tail_int);
	cudaMalloc(&d_t_force, bytes_tail3);

	cudaMalloc(&d_LH_g_pos, bytes_lh_g);
        cudaMalloc(&d_LH_conn, bytes_lh_conn);
        cudaMalloc(&d_nc_lh_flag, bytes_nc_int);
        cudaMalloc(&d_beta_lh, bytes_LH);
        cudaMalloc(&d_r_lh, bytes_LH3);
        cudaMalloc(&d_LH_q, bytes_lh_d);
        cudaMalloc(&d_LH_vdw_hh, bytes_lh_d);
        cudaMalloc(&d_LH_vdw_hc, bytes_lh_d);
        cudaMalloc(&d_LH_vdw_hl, bytes_lh_d);
        cudaMalloc(&d_LH_vdw_ht, bytes_lh_d);
        cudaMalloc(&d_LH_kstr, bytes_lh_d);
        cudaMalloc(&d_LH_kben, bytes_lh_d);
        cudaMalloc(&d_LH_streq, bytes_lh_d);
        cudaMalloc(&d_LH_betaeq, bytes_lh_d);
        cudaMalloc(&d_LH_force, bytes_LH3);

	cudaMalloc(&d_force_m, bytes_r_t_lh_m);
	cudaMalloc(&d_torque_m, bytes_r_t_lh_m);
	cudaMalloc(&d_t_force_m, bytes_r_t_lh_m);
	cudaMalloc(&d_LH_force_m, bytes_r_t_lh_m);
	cudaMalloc(&d_temp_force_m, bytes_r_t_lh_m);

	//Copy data to the device

	cudaMemcpy(d_Energy, Energy, bytes,cudaMemcpyHostToDevice);
	cudaMemcpy(d_type, type, bytes_ni, cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, r, bytes_n3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_a, a, bytes_n3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, bytes_n3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, bytes_n3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_alpha, alpha, bytes_nd, cudaMemcpyHostToDevice);
	cudaMemcpy(d_beta, beta, bytes_nd, cudaMemcpyHostToDevice);
	cudaMemcpy(d_gamma, gamma, bytes_nd, cudaMemcpyHostToDevice);
	cudaMemcpy(d_length, length, bytes_nd, cudaMemcpyHostToDevice);
	cudaMemcpy(d_a_dna, a_dna, bytes_nc3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b_dna, b_dna, bytes_nc3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c_dna, c_dna, bytes_nc3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_alpha_p, alpha_p, bytes_nc, cudaMemcpyHostToDevice);
	cudaMemcpy(d_beta_p, beta_p, bytes_nc, cudaMemcpyHostToDevice);
	cudaMemcpy(d_gamma_p, gamma_p, bytes_nc, cudaMemcpyHostToDevice);
	cudaMemcpy(d_phi_o, phi_o, bytes_nd, cudaMemcpyHostToDevice);
	cudaMemcpy(d_core_pos, core_pos, bytes_Nq3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_core_q, core_q, bytes_Nq, cudaMemcpyHostToDevice);
	cudaMemcpy(d_force, force, bytes_n3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_torque, torque, bytes_n3, cudaMemcpyHostToDevice);

	cudaMemcpy(d_tail_pos, tail_pos, bytes_t3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tail_fix, tail_fix, bytes_t, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nc_t_flag, nc_t_flag, bytes_nc_int, cudaMemcpyHostToDevice);
	cudaMemcpy(d_r_t, r_t, bytes_tail3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_beta_t, beta_t, bytes_tail, cudaMemcpyHostToDevice);
	cudaMemcpy(d_h_t, h_t, bytes_tail, cudaMemcpyHostToDevice);
	cudaMemcpy(d_g_t, g_t, bytes_tail, cudaMemcpyHostToDevice);
	cudaMemcpy(d_lo_t, lo_t, bytes_tail, cudaMemcpyHostToDevice);
	cudaMemcpy(d_beta_o_t, beta_o_t, bytes_tail, cudaMemcpyHostToDevice);
	cudaMemcpy(d_t_q, t_q, bytes_tail, cudaMemcpyHostToDevice);
	cudaMemcpy(d_t_rad, t_rad, bytes_tail, cudaMemcpyHostToDevice);
	cudaMemcpy(d_t_grp, t_grp, bytes_tail_int, cudaMemcpyHostToDevice);
	cudaMemcpy(d_t_fix, t_fix, bytes_tail_int, cudaMemcpyHostToDevice);
	cudaMemcpy(d_t_force, t_force, bytes_tail3, cudaMemcpyHostToDevice);

	cudaMemcpy(d_LH_g_pos, LH_g_pos, bytes_lh_g, cudaMemcpyHostToDevice);
        cudaMemcpy(d_LH_conn, LH_conn, bytes_lh_conn, cudaMemcpyHostToDevice);
        cudaMemcpy(d_nc_lh_flag, nc_lh_flag, bytes_nc_int, cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta_lh, beta_lh, bytes_LH, cudaMemcpyHostToDevice);
        cudaMemcpy(d_r_lh, r_lh, bytes_LH3, cudaMemcpyHostToDevice);
        cudaMemcpy(d_LH_q, LH_q, bytes_lh_d, cudaMemcpyHostToDevice);
        cudaMemcpy(d_LH_vdw_hh, LH_vdw_hh, bytes_lh_d, cudaMemcpyHostToDevice);
        cudaMemcpy(d_LH_vdw_hc, LH_vdw_hc, bytes_lh_d, cudaMemcpyHostToDevice);
        cudaMemcpy(d_LH_vdw_hl, LH_vdw_hl, bytes_lh_d, cudaMemcpyHostToDevice);
        cudaMemcpy(d_LH_vdw_ht, LH_vdw_ht, bytes_lh_d, cudaMemcpyHostToDevice);
        cudaMemcpy(d_LH_kstr, LH_kstr, bytes_lh_d, cudaMemcpyHostToDevice);
        cudaMemcpy(d_LH_kben, LH_kben, bytes_lh_d, cudaMemcpyHostToDevice);
        cudaMemcpy(d_LH_streq, LH_streq, bytes_lh_d, cudaMemcpyHostToDevice);
        cudaMemcpy(d_LH_betaeq, LH_betaeq, bytes_lh_d, cudaMemcpyHostToDevice);
        cudaMemcpy(d_LH_force, LH_force, bytes_LH3, cudaMemcpyHostToDevice);


	int threads_per_block_rtlh = 16;
        dim3 block_size_rtlh(threads_per_block_rtlh, threads_per_block_rtlh);
        dim3 grid_size_rtlh(maxrtlh*3 / block_size_rtlh.x + 1 , maxrtlh*3 / block_size_rtlh.y + 1 );

	init_force_torque_m<<<grid_size_rtlh, block_size_rtlh>>>(maxrtlh, d_force_m, d_torque_m, d_t_force_m, d_LH_force_m, d_temp_force_m);


	int num_thread = 64;
        int num_block = (n + num_thread -1) / num_thread;

	mechanical_force_and_torque_cuda <<<num_block, num_thread>>> (maxrtlh, n_c, nc3, n, n3, d_type, d_r, d_a, d_b, d_c, d_alpha, d_beta, d_gamma, d_length, d_a_dna, d_b_dna, d_c_dna, d_alpha_p, d_beta_p, d_gamma_p, h, g, s, d_phi_o, d_force, d_torque, d_Energy, d_force_m, d_torque_m);
	
	//assign number of blocks and threads

        int threads_per_block = 16;
        dim3 block_size(threads_per_block, threads_per_block);
        dim3 grid_size(n / block_size.x + 1 , n / block_size.y + 1 );

	Electrostatic_and_Excluded_volume_force_cuda <<<grid_size, block_size>>> (maxrtlh, n, n3, n_c, nc3, d_type, d_r, d_a, d_b, d_c, debyell, debye, q_l, k_e, k_ex, k_h1, sigma_DNA_DNA, sigma_DNA_Core, sigma_Core_Core, Nq, Nq3, d_core_pos, d_core_q, d_force, d_torque, d_Energy, d_force_m, d_torque_m);
	
	//assign number of blocks and threads

	if (n_tail >0){
		int num_thread_t = 64;
        	int num_block_t = (n_tail + num_thread_t -1) / num_thread_t;

		tail_force_cuda_str_ben<<<num_block_t, num_thread_t>>>(maxrtlh, n, n_t, d_type, n_tail, n_tail3, d_tail_pos, d_tail_fix, d_nc_t_flag, d_r_t, d_beta_t, d_h_t, d_g_t, d_lo_t, d_beta_o_t, h, k_e, debye, k_ex, d_t_q, d_t_rad, d_t_grp, d_t_fix, d_r, d_a, d_b, d_c, q_l, Nq, Nq3, d_core_pos, d_core_q, d_t_force, d_force, d_torque, sigma_Tail_Tail, sigma_Tail_Linker, sigma_Tail_Core, d_Energy, d_t_force_m);

		int threads_per_block_t = 16;
        	dim3 block_size_t(threads_per_block_t, threads_per_block_t);
        	dim3 grid_size_t(n_tail / block_size_t.x + 1 , n_tail / block_size_t.y + 1 );

		tail_force_cuda_ele_vdw<<<grid_size_t, block_size_t>>>(maxrtlh, n, n_t, d_type, n_tail, n_tail3, d_tail_pos, d_tail_fix, d_nc_t_flag, d_r_t, d_beta_t, d_h_t, d_g_t, d_lo_t, d_beta_o_t, h, k_e, debye, k_ex, d_t_q, d_t_rad, d_t_grp, d_t_fix, d_r, d_a, d_b, d_c, q_l, Nq, Nq3, d_core_pos, d_core_q, d_t_force, d_force, d_torque, sigma_Tail_Tail, sigma_Tail_Linker, sigma_Tail_Core, d_Energy, d_t_force_m, d_force_m, d_torque_m);

	}

	//assign number of blocks and threads

	if (n_LH>0){
        	int num_thread_lh = 64;
        	int num_block_lh = (n_LH + num_thread_lh -1) / num_thread_lh;


		Linker_Histone_str_ben_cuda<<<num_block_lh, num_thread_lh>>>(maxrtlh, n, n_tail, n_lh_n, n_lh_g, n_lh_c, d_t_grp, d_t_fix, d_type, n_LH, n_LH3, d_LH_g_pos, d_LH_conn, d_nc_lh_flag, d_beta_lh, d_r_lh, d_LH_q, k_e, debye, k_ex, d_LH_vdw_hh, d_LH_vdw_hc, d_LH_vdw_hl, d_LH_vdw_ht, d_LH_kstr, d_LH_kben, d_LH_streq, d_LH_betaeq, d_r, d_a, d_b, d_c, q_l, Nq, Nq3, d_core_pos, d_core_q, d_r_t, d_t_q, d_t_force, d_LH_force, d_force, d_torque, d_Energy, d_LH_force_m);

		int threads_per_block_lh = 16;
        	dim3 block_size_lh(threads_per_block_lh, threads_per_block_lh);
        	dim3 grid_size_lh(maxrtlh / block_size_lh.x + 1 , maxrtlh / block_size_lh.y + 1 );

		Linker_Histone_ele_vdw_cuda<<<grid_size_lh, block_size_lh>>>(maxrtlh, n, n_tail, n_lh_n, n_lh_g, n_lh_c, d_t_grp, d_t_fix, d_type, n_LH, n_LH3, d_LH_g_pos, d_LH_conn, d_nc_lh_flag, d_beta_lh, d_r_lh, d_LH_q, k_e, debye, k_ex, d_LH_vdw_hh, d_LH_vdw_hc, d_LH_vdw_hl, d_LH_vdw_ht, d_LH_kstr, d_LH_kben, d_LH_streq, d_LH_betaeq, d_r, d_a, d_b, d_c, q_l, Nq, Nq3, d_core_pos, d_core_q, d_r_t, d_t_q, d_t_force, d_LH_force, d_force, d_torque, d_Energy, d_LH_force_m, d_force_m, d_torque_m, d_t_force_m, d_temp_force_m);

	}

	int num_thread_rtlh = 64;
        int num_block_rtlh = (maxrtlh + num_thread_rtlh -1) / num_thread_rtlh;

	temp_force_2_force_m <<<num_block_rtlh, num_thread_rtlh>>> (n, maxrtlh, d_t_grp, d_type, d_temp_force_m, d_force_m, d_torque_m);
	force_reduction <<<num_block_rtlh, num_thread_rtlh>>> (n, n_tail, n_LH, maxrtlh, maxrtlh*3, d_force_m, d_force, d_torque_m, d_torque, d_t_force_m, d_t_force, d_LH_force_m, d_LH_force);

	cudaMemcpy(force, d_force, bytes_n3, cudaMemcpyDeviceToHost);
	cudaMemcpy(torque, d_torque, bytes_n3, cudaMemcpyDeviceToHost);
	cudaMemcpy(t_force, d_t_force, bytes_tail3, cudaMemcpyDeviceToHost);
	cudaMemcpy(LH_force, d_LH_force, bytes_LH3, cudaMemcpyDeviceToHost);
	cudaMemcpy(Energy, d_Energy, bytes, cudaMemcpyDeviceToHost);

	cudaFree(d_Energy);
	cudaFree(d_type);
	cudaFree(d_r);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_alpha);
	cudaFree(d_beta);
	cudaFree(d_gamma);
	cudaFree(d_length);
	cudaFree(d_a_dna);
	cudaFree(d_b_dna);
	cudaFree(d_c_dna);
	cudaFree(d_alpha_p);
	cudaFree(d_beta_p);
	cudaFree(d_gamma_p);
	cudaFree(d_phi_o);
	cudaFree(d_force);
	cudaFree(d_torque);
	cudaFree(d_core_pos);
        cudaFree(d_core_q);

	cudaFree(d_tail_pos);
	cudaFree(d_tail_fix);
	cudaFree(d_nc_t_flag);
	cudaFree(d_r_t);
	cudaFree(d_beta_t);
	cudaFree(d_h_t);
	cudaFree(d_g_t);
	cudaFree(d_lo_t);
	cudaFree(d_beta_o_t);
	cudaFree(d_t_q);
	cudaFree(d_t_rad);
	cudaFree(d_t_grp);
	cudaFree(d_t_fix);
	cudaFree(d_t_force);

	cudaFree(d_LH_g_pos);
        cudaFree(d_LH_conn);
        cudaFree(d_nc_lh_flag);
        cudaFree(d_beta_lh);
        cudaFree(d_r_lh);
        cudaFree(d_LH_q);
        cudaFree(d_LH_vdw_hh);
        cudaFree(d_LH_vdw_hc);
        cudaFree(d_LH_vdw_hl);
        cudaFree(d_LH_vdw_ht);
        cudaFree(d_LH_kstr);
        cudaFree(d_LH_kben);
        cudaFree(d_LH_streq);
        cudaFree(d_LH_betaeq);
        cudaFree(d_LH_force);

	cudaFree(d_force_m);
	cudaFree(d_torque_m);
	cudaFree(d_t_force_m);
	cudaFree(d_LH_force_m);
	cudaFree(d_temp_force_m);
}
*/

extern "C++" void main_cuda(int n_c, int nc3, int step, int number_of_steps, double time_step, double del, int frequency_RP, int frequency_of_sampling, double h, double g, double s, double debyell, double debye, double q_l, double k_e, double k_ex, double k_h1, double sigma_DNA_DNA, double sigma_DNA_Core, double sigma_Core_Core, double sigma_Tail_Tail, double sigma_Tail_Linker, double sigma_Tail_Core, int Nq, int Nq3, int n_t, int n_lh_n, int n_lh_g, int n_lh_c, int n, int n3, int n_tail, int n_tail3, int n_LH, int n_LH3, double a1, double a2, double s2dt, double* rr, double* p, double* Energy, double* h_r, double* h_a, double* h_b, double* h_c, double* h_r_t, double* h_r_lh, double* h_rad_all, int* nc_lh_flag){

	cudaFree(d_r_lh);
	cudaFree(d_r_lh_n);
	cudaFree(d_beta_lh);
	cudaFree(d_beta_lh_n);
	cudaFree(d_LH_force);
	cudaFree(d_LH_force_n);
	cudaFree(d_r_all);
	cudaFree(d_rad_all);
	cudaFree(d_LH_force_tmp);
	cudaFree(d_nc_lh_flag);

	int maxrtlh;
	int n_D, n_D3;
	n_D = n + n_tail + n_LH;
	n_D3 = n_D*3;

        if (n>n_LH){
                if (n>n_tail){
                        maxrtlh=n;
                }else{
                        maxrtlh=n_tail;
                }
        }else{
                if (n_LH>n_tail){
                        maxrtlh=n_LH;
                }else{
                        maxrtlh=n_tail;
                }
        }

	size_t bytes = sizeof(double);

	size_t bytes_LH = n_LH*sizeof(double);
        size_t bytes_LH3 = n_LH3*sizeof(double);

        size_t bytes_n_D = n_D*sizeof(double);
        size_t bytes_n_D3 = n_D3*sizeof(double);
	size_t bytes_nc_int = n_c*sizeof(int);

	cudaMalloc(&d_LH_force, bytes_LH3);
        cudaMalloc(&d_LH_force_n, bytes_LH3);

        cudaMalloc(&d_LH_force_tmp, bytes_LH3);

        cudaMalloc(&d_beta_lh_n, bytes_LH);
        cudaMalloc(&d_r_lh_n, bytes_LH3);

        cudaMalloc(&d_r_all, bytes_n_D3);
        cudaMalloc(&d_rad_all, bytes_n_D);
        cudaMalloc(&d_r_lh, bytes_LH3);
	cudaMalloc(&d_beta_lh, bytes_LH);

	cudaMalloc(&d_nc_lh_flag, bytes_nc_int);

	cudaMemcpy(d_r_lh, h_r_lh, bytes_LH3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rad_all, h_rad_all, bytes_n_D, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nc_lh_flag, nc_lh_flag, bytes_nc_int, cudaMemcpyHostToDevice);

	int num_thread_lh = 64;
        int num_block_lh = (n_LH + num_thread_lh -1) / num_thread_lh;

	update_LH_beta_cuda <<<num_block_lh, num_thread_lh>>> (n_LH, n_lh_c, d_r_lh, d_beta_lh);

	int num_thread_rtlh3 = 64;
        int num_block_rtlh3 = (maxrtlh*3 + num_thread_rtlh3 -1) / num_thread_rtlh3;

	step_init<<<num_block_rtlh3, num_thread_rtlh3>>>(n, n_tail, n_LH, d_force, d_force_n, d_torque, d_torque_n, d_t_force, d_t_force_n, d_LH_force, d_LH_force_n, d_Energy, d_r_all, d_r, d_r_t, d_r_lh);	

//	if (step%frequency_RP == 0){
	cudaFree(d_D);
        cudaFree(d_Chol);

	size_t bytes_D = n_D3*n_D3*sizeof(double);
	
	cudaMalloc(&d_D, bytes_D);
        cudaMalloc(&d_Chol, bytes_D);	

	cusolverDnHandle_t solver_handle;
	cusolverDnCreate(&solver_handle);

	int work_size = 0;
	int *devInfo;
	cudaMalloc(&devInfo, sizeof(int));

	int threads_per_block_D = 32;
        dim3 block_size_D(threads_per_block_D, threads_per_block_D);
	dim3 grid_size_D(n_D3 / block_size_D.x + 1 , n_D3 / block_size_D.y + 1 );

	Diffusion_Tensor_CUDA <<<grid_size_D, block_size_D>>> (n_D, n_D3, d_r_all, a1, a2, d_rad_all, d_D, d_Chol);

	cusolverDnDpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_UPPER, n_D3, d_Chol, n_D3, &work_size);
	double *work;
	cudaMalloc(&work, work_size * sizeof(double));
        cusolverDnDpotrf(solver_handle, CUBLAS_FILL_MODE_UPPER, n_D3, d_Chol, n_D3, work, work_size, devInfo);
	Cholesky_Decomposition_mod <<<grid_size_D, block_size_D>>> (n_D3, d_Chol);
	cudaFree(devInfo);
	cudaFree(work);

        cusolverDnDestroy(solver_handle);
//        }

	double* d_p;
	double* d_rr;

	size_t bytes_p = n_D3*sizeof(double);
	size_t bytes_rr = n3*sizeof(double);

	cudaMalloc(&d_p, bytes_p);
	cudaMalloc(&d_rr, bytes_rr);

	cudaMemcpy(d_p, p, bytes_p, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rr, rr, bytes_rr, cudaMemcpyHostToDevice);

	int num_thread_rd = 32;
        int num_block_rd = (n_D3 + num_thread_rd -1) / num_thread_rd;

	rd_cal <<<num_block_rd, num_thread_rd>>> (n_D3, d_rd, d_Chol, d_p, s2dt);

	//Force and torque calculation

	int threads_per_block_rtlh = 16;
        dim3 block_size_rtlh(threads_per_block_rtlh, threads_per_block_rtlh);
        dim3 grid_size_rtlh(maxrtlh*3 / block_size_rtlh.x + 1 , maxrtlh*3 / block_size_rtlh.y + 1 );

        init_force_torque_m<<<grid_size_rtlh, block_size_rtlh>>>(maxrtlh, d_force_m, d_torque_m, d_t_force_m, d_LH_force_m, d_temp_force_m, d_Energy_m);


        int num_thread = 64;
        int num_block = (n + num_thread -1) / num_thread;

        mechanical_force_and_torque_cuda <<<num_block, num_thread>>> (maxrtlh, n_c, nc3, n, n3, d_type, d_r, d_a, d_b, d_c, d_alpha, d_beta, d_gamma, d_length, d_a_dna, d_b_dna, d_c_dna, d_alpha_p, d_beta_p, d_gamma_p, h, g, s, d_phi_o, d_force, d_torque, d_Energy, d_force_m, d_torque_m, d_Energy_m);

        int threads_per_block = 16;
        dim3 block_size(threads_per_block, threads_per_block);
        dim3 grid_size(n / block_size.x + 1 , n / block_size.y + 1 );

        Electrostatic_and_Excluded_volume_force_cuda <<<grid_size, block_size>>> (maxrtlh, n, n3, n_c, nc3, d_type, d_r, d_a, d_b, d_c, debyell, debye, q_l, k_e, k_ex, k_h1, sigma_DNA_DNA, sigma_DNA_Core, sigma_Core_Core, Nq, Nq3, d_core_pos, d_core_q, d_force, d_torque, d_Energy, d_force_m, d_torque_m, d_Energy_m);

	int num_thread_t = 64;
        int num_block_t = (n_tail + num_thread_t -1) / num_thread_t;

	int threads_per_block_t = 16;
        dim3 block_size_t(threads_per_block_t, threads_per_block_t);
        dim3 grid_size_t(n_tail / block_size_t.x + 1 , n_tail / block_size_t.y + 1 );

        if (n_tail >0){

                tail_force_cuda_str_ben<<<num_block_t, num_thread_t>>>(maxrtlh, n, n_t, d_type, n_tail, n_tail3, d_tail_pos, d_tail_fix, d_nc_t_flag, d_r_t, d_beta_t, d_h_t, d_g_t, d_lo_t, d_beta_o_t, h, k_e, debye, k_ex, d_t_q, d_t_rad, d_t_grp, d_t_fix, d_r, d_a, d_b, d_c, q_l, Nq, Nq3, d_core_pos, d_core_q, d_t_force, d_force, d_torque, sigma_Tail_Tail, sigma_Tail_Linker, sigma_Tail_Core, d_Energy, d_t_force_m, d_Energy_m);


                tail_force_cuda_ele_vdw<<<grid_size_t, block_size_t>>>(maxrtlh, n, n_t, d_type, n_tail, n_tail3, d_tail_pos, d_tail_fix, d_nc_t_flag, d_r_t, d_beta_t, d_h_t, d_g_t, d_lo_t, d_beta_o_t, h, k_e, debye, k_ex, d_t_q, d_t_rad, d_t_grp, d_t_fix, d_r, d_a, d_b, d_c, q_l, Nq, Nq3, d_core_pos, d_core_q, d_t_force, d_force, d_torque, sigma_Tail_Tail, sigma_Tail_Linker, sigma_Tail_Core, d_Energy, d_t_force_m, d_force_m, d_torque_m, d_Energy_m);

        }

//	int num_thread_lh = 64;
//        int num_block_lh = (n_LH + num_thread_lh -1) / num_thread_lh;

	int threads_per_block_lh = 16;
        dim3 block_size_lh(threads_per_block_lh, threads_per_block_lh);
        dim3 grid_size_lh(maxrtlh / block_size_lh.x + 1 , maxrtlh / block_size_lh.y + 1 );

        if (n_LH>0){


                Linker_Histone_str_ben_cuda<<<num_block_lh, num_thread_lh>>>(maxrtlh, n, n_tail, n_lh_n, n_lh_g, n_lh_c, d_t_grp, d_t_fix, d_type, n_LH, n_LH3, d_LH_g_pos, d_LH_conn, d_nc_lh_flag, d_beta_lh, d_r_lh, d_LH_q, k_e, debye, k_ex, d_LH_vdw_hh, d_LH_vdw_hc, d_LH_vdw_hl, d_LH_vdw_ht, d_LH_kstr, d_LH_kben, d_LH_streq, d_LH_betaeq, d_r, d_a, d_b, d_c, q_l, Nq, Nq3, d_core_pos, d_core_q, d_r_t, d_t_q, d_t_force, d_LH_force, d_force, d_torque, d_Energy, d_LH_force_m, d_Energy_m);


                Linker_Histone_ele_vdw_cuda<<<grid_size_lh, block_size_lh>>>(maxrtlh, n, n_tail, n_lh_n, n_lh_g, n_lh_c, d_t_grp, d_t_fix, d_type, n_LH, n_LH3, d_LH_g_pos, d_LH_conn, d_nc_lh_flag, d_beta_lh, d_r_lh, d_LH_q, k_e, debye, k_ex, d_LH_vdw_hh, d_LH_vdw_hc, d_LH_vdw_hl, d_LH_vdw_ht, d_LH_kstr, d_LH_kben, d_LH_streq, d_LH_betaeq, d_r, d_a, d_b, d_c, q_l, Nq, Nq3, d_core_pos, d_core_q, d_r_t, d_t_q, d_t_force, d_LH_force, d_force, d_torque, d_Energy, d_LH_force_m, d_force_m, d_torque_m, d_t_force_m, d_temp_force_m, d_Energy_m);

        }

        int num_thread_rtlh = 64;
        int num_block_rtlh = (maxrtlh + num_thread_rtlh -1) / num_thread_rtlh;

        temp_force_2_force_m <<<num_block_rtlh, num_thread_rtlh>>> (n, maxrtlh, d_t_grp, d_type, d_temp_force_m, d_force_m, d_torque_m);
        force_reduction <<<num_block_rtlh, num_thread_rtlh>>> (n, n_tail, n_LH, maxrtlh, maxrtlh*3, d_force_m, d_force, d_torque_m, d_torque, d_t_force_m, d_t_force, d_LH_force_m, d_LH_force);

	Energy_reduction  <<<num_block_rtlh, num_thread_rtlh>>> (maxrtlh, d_Energy, d_Energy_m);

	cudaMemcpy(Energy, d_Energy, bytes, cudaMemcpyDeviceToHost);

	rotation_cal <<<num_block, num_thread>>> (n, d_d_theta, d_type, time_step, d_torque, d_rr, d_a, d_b, d_c, d_a_n, d_b_n, d_c_n);

	translation_cal <<<num_block_rd, num_thread_rd>>> (n_D3, n3, n_tail3, n_LH3, d_r, d_r_n, d_rd, d_r_t, d_r_t_n, d_r_lh, d_r_lh_n, del, d_force, d_t_force, d_LH_force, d_D);

	update_tail_beta_cuda <<<num_block_t, num_thread_t>>> (n_tail, d_r_t_n, d_beta_t_n, d_t_grp);

	update_LH_beta_cuda <<<num_block_lh, num_thread_lh>>> (n_LH, n_lh_c, d_r_lh_n, d_beta_lh_n);
	
	update_Euler_Angle_cuda <<<num_block, num_thread>>> (n_c, nc3, n, n3, d_type, d_r_n, d_a_n, d_b_n, d_c_n, d_alpha_n, d_beta_n, d_gamma_n, d_length_n, d_a_dna_n, d_b_dna_n, d_c_dna_n, d_alpha_p_n, d_beta_p_n, d_gamma_p_n);

	
	init_force_torque_m<<<grid_size_rtlh, block_size_rtlh>>>(maxrtlh, d_force_m, d_torque_m, d_t_force_m, d_LH_force_m, d_temp_force_m, d_Energy_m);

	mechanical_force_and_torque_cuda <<<num_block, num_thread>>> (maxrtlh, n_c, nc3, n, n3, d_type, d_r_n, d_a_n, d_b_n, d_c_n, d_alpha_n, d_beta_n, d_gamma_n, d_length_n, d_a_dna_n, d_b_dna_n, d_c_dna_n, d_alpha_p_n, d_beta_p_n, d_gamma_p_n, h, g, s, d_phi_o, d_force_n, d_torque_n, d_Energy, d_force_m, d_torque_m, d_Energy_m);

	Electrostatic_and_Excluded_volume_force_cuda <<<grid_size, block_size>>> (maxrtlh, n, n3, n_c, nc3, d_type, d_r_n, d_a_n, d_b_n, d_c_n, debyell, debye, q_l, k_e, k_ex, k_h1, sigma_DNA_DNA, sigma_DNA_Core, sigma_Core_Core, Nq, Nq3, d_core_pos, d_core_q, d_force_n, d_torque_n, d_Energy, d_force_m, d_torque_m, d_Energy_m);

	if (n_tail >0){
		tail_force_cuda_str_ben<<<num_block_t, num_thread_t>>>(maxrtlh, n, n_t, d_type, n_tail, n_tail3, d_tail_pos, d_tail_fix, d_nc_t_flag, d_r_t_n, d_beta_t_n, d_h_t, d_g_t, d_lo_t, d_beta_o_t, h, k_e, debye, k_ex, d_t_q, d_t_rad, d_t_grp, d_t_fix, d_r_n, d_a_n, d_b_n, d_c_n, q_l, Nq, Nq3, d_core_pos, d_core_q, d_t_force_n, d_force_n, d_torque_n, sigma_Tail_Tail, sigma_Tail_Linker, sigma_Tail_Core, d_Energy, d_t_force_m, d_Energy_m);
		tail_force_cuda_ele_vdw<<<grid_size_t, block_size_t>>>(maxrtlh, n, n_t, d_type, n_tail, n_tail3, d_tail_pos, d_tail_fix, d_nc_t_flag, d_r_t_n, d_beta_t_n, d_h_t, d_g_t, d_lo_t, d_beta_o_t, h, k_e, debye, k_ex, d_t_q, d_t_rad, d_t_grp, d_t_fix, d_r_n, d_a_n, d_b_n, d_c_n, q_l, Nq, Nq3, d_core_pos, d_core_q, d_t_force_n, d_force_n, d_torque_n, sigma_Tail_Tail, sigma_Tail_Linker, sigma_Tail_Core, d_Energy, d_t_force_m, d_force_m, d_torque_m, d_Energy_m);

        }

	if (n_LH>0){
		Linker_Histone_str_ben_cuda<<<num_block_lh, num_thread_lh>>>(maxrtlh, n, n_tail, n_lh_n, n_lh_g, n_lh_c, d_t_grp, d_t_fix, d_type, n_LH, n_LH3, d_LH_g_pos, d_LH_conn, d_nc_lh_flag, d_beta_lh_n, d_r_lh_n, d_LH_q, k_e, debye, k_ex, d_LH_vdw_hh, d_LH_vdw_hc, d_LH_vdw_hl, d_LH_vdw_ht, d_LH_kstr, d_LH_kben, d_LH_streq, d_LH_betaeq, d_r_n, d_a_n, d_b_n, d_c_n, q_l, Nq, Nq3, d_core_pos, d_core_q, d_r_t_n, d_t_q, d_t_force_n, d_LH_force_n, d_force_n, d_torque_n, d_Energy, d_LH_force_m, d_Energy_m);
		Linker_Histone_ele_vdw_cuda<<<grid_size_lh, block_size_lh>>>(maxrtlh, n, n_tail, n_lh_n, n_lh_g, n_lh_c, d_t_grp, d_t_fix, d_type, n_LH, n_LH3, d_LH_g_pos, d_LH_conn, d_nc_lh_flag, d_beta_lh_n, d_r_lh_n, d_LH_q, k_e, debye, k_ex, d_LH_vdw_hh, d_LH_vdw_hc, d_LH_vdw_hl, d_LH_vdw_ht, d_LH_kstr, d_LH_kben, d_LH_streq, d_LH_betaeq, d_r_n, d_a_n, d_b_n, d_c_n, q_l, Nq, Nq3, d_core_pos, d_core_q, d_r_t_n, d_t_q, d_t_force_n, d_LH_force_n, d_force_n, d_torque_n, d_Energy, d_LH_force_m, d_force_m, d_torque_m, d_t_force_m, d_temp_force_m, d_Energy_m);
        }

	temp_force_2_force_m <<<num_block_rtlh, num_thread_rtlh>>> (n, maxrtlh, d_t_grp, d_type, d_temp_force_m, d_force_m, d_torque_m);
        force_reduction <<<num_block_rtlh, num_thread_rtlh>>> (n, n_tail, n_LH, maxrtlh, maxrtlh*3, d_force_m, d_force_n, d_torque_m, d_torque_n, d_t_force_m, d_t_force_n, d_LH_force_m, d_LH_force_n);

	force_torque_tmp<<<num_block_rtlh3, num_thread_rtlh3>>>(n3, n_tail3, n_LH3, d_force, d_torque, d_t_force, d_LH_force, d_force_n, d_torque_n, d_t_force_n, d_LH_force_n, d_force_tmp, d_torque_tmp, d_t_force_tmp, d_LH_force_tmp);

	rotation_cal <<<num_block, num_thread>>> (n, d_d_theta, d_type, time_step, d_torque_tmp, d_rr, d_a, d_b, d_c, d_a_n, d_b_n, d_c_n);

        translation_cal <<<num_block_rd, num_thread_rd>>> (n_D3, n3, n_tail3, n_LH3, d_r, d_r_n, d_rd, d_r_t, d_r_t_n, d_r_lh, d_r_lh_n, del, d_force_tmp, d_t_force_tmp, d_LH_force_tmp, d_D);

	final_updates <<<num_block_rtlh3, num_thread_rtlh3>>> (n3, n_tail3, n_LH3, d_r, d_r_n, d_a, d_a_n, d_b, d_b_n, d_c, d_c_n, d_r_t, d_r_t_n, d_r_lh, d_r_lh_n);

	update_Euler_Angle_cuda <<<num_block, num_thread>>> (n_c, nc3, n, n3, d_type, d_r, d_a, d_b, d_c, d_alpha, d_beta, d_gamma, d_length, d_a_dna, d_b_dna, d_c_dna, d_alpha_p, d_beta_p, d_gamma_p);

	update_tail_beta_cuda <<<num_block_t, num_thread_t>>> (n_tail, d_r_t, d_beta_t, d_t_grp);

	update_LH_beta_cuda <<<num_block_lh, num_thread_lh>>> (n_LH, n_lh_c, d_r_lh, d_beta_lh);

	size_t bytes_r_lh = n_LH3*sizeof(double);
	cudaMemcpy(h_r_lh, d_r_lh, bytes_r_lh, cudaMemcpyDeviceToHost);

	if (step%frequency_of_sampling == 0 or step == number_of_steps-1){
		size_t bytes_r = n3*sizeof(double);		
		size_t bytes_r_t = n_tail3*sizeof(double);

		cudaMemcpy(h_r, d_r, bytes_r, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_a, d_a, bytes_r, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_b, d_b, bytes_r, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_c, d_c, bytes_r, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_r_t, d_r_t, bytes_r_t, cudaMemcpyDeviceToHost);
	}



	cudaFree(d_p);
        cudaFree(d_rr);
}

extern "C++" void free_all(){

	cudaFree(d_Energy);
        cudaFree(d_type);
        cudaFree(d_r);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaFree(d_alpha);
        cudaFree(d_beta);
        cudaFree(d_gamma);
        cudaFree(d_length);
        cudaFree(d_a_dna);
        cudaFree(d_b_dna);
        cudaFree(d_c_dna);
        cudaFree(d_alpha_p);
        cudaFree(d_beta_p);
        cudaFree(d_gamma_p);
        cudaFree(d_phi_o);
        cudaFree(d_force);
        cudaFree(d_torque);
        cudaFree(d_core_pos);
        cudaFree(d_core_q);

	cudaFree(d_r_n);
        cudaFree(d_a_n);
        cudaFree(d_b_n);
        cudaFree(d_c_n);
        cudaFree(d_alpha_n);
        cudaFree(d_beta_n);
        cudaFree(d_gamma_n);
        cudaFree(d_length_n);
        cudaFree(d_a_dna_n);
        cudaFree(d_b_dna_n);
        cudaFree(d_c_dna_n);
        cudaFree(d_alpha_p_n);
        cudaFree(d_beta_p_n);
        cudaFree(d_gamma_p_n);

	cudaFree(d_force_n);
        cudaFree(d_torque_n);
	cudaFree(d_force_tmp);
        cudaFree(d_torque_tmp);

	cudaFree(d_r_all);
	cudaFree(d_rad_all);
	cudaFree(d_d_theta);
	cudaFree(d_rd);

        cudaFree(d_tail_pos);
        cudaFree(d_tail_fix);
        cudaFree(d_nc_t_flag);
        cudaFree(d_r_t);
        cudaFree(d_beta_t);
        cudaFree(d_h_t);
        cudaFree(d_g_t);
        cudaFree(d_lo_t);
        cudaFree(d_beta_o_t);
        cudaFree(d_t_q);
        cudaFree(d_t_rad);
        cudaFree(d_t_grp);
        cudaFree(d_t_fix);
        cudaFree(d_t_force);

	cudaFree(d_r_t_n);
        cudaFree(d_beta_t_n);

	cudaFree(d_t_force_n);
	cudaFree(d_t_force_tmp);

        cudaFree(d_LH_g_pos);
        cudaFree(d_LH_conn);
        cudaFree(d_nc_lh_flag);
        cudaFree(d_beta_lh);
        cudaFree(d_r_lh);
        cudaFree(d_LH_q);
        cudaFree(d_LH_vdw_hh);
        cudaFree(d_LH_vdw_hc);
        cudaFree(d_LH_vdw_hl);
        cudaFree(d_LH_vdw_ht);
        cudaFree(d_LH_kstr);
        cudaFree(d_LH_kben);
        cudaFree(d_LH_streq);
        cudaFree(d_LH_betaeq);
        cudaFree(d_LH_force);

	cudaFree(d_beta_lh_n);
        cudaFree(d_r_lh_n);

	cudaFree(d_LH_force_n);
	cudaFree(d_LH_force_tmp);

        cudaFree(d_force_m);
        cudaFree(d_torque_m);
        cudaFree(d_t_force_m);
        cudaFree(d_LH_force_m);
        cudaFree(d_temp_force_m);

	cudaFree(d_D);
	cudaFree(d_Chol);

	cudaFree(d_Energy_m);
}
