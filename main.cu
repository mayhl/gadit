
// ----------------------------------------------------------------------------------
// Copyright 2016-2017 Michael-Angelo Yick-Hang Lam
//
// The development of this software was supported by the National Science Foundation
// (NSF) Grant Number DMS-1211713.
//
// This file is part of GADIT.
//
// GADIT is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 3 as published by
// the Free Software Foundation.
//
// GADIT is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with GADIT.  If not, see <http://www.gnu.org/licenses/>.
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
// Name:			main.cu
// Version: 		1.0
// Purpose:			Minimal example code of how to execute GADIT.
// ----------------------------------------------------------------------------------

#include "gadit_solver.h"

int main(int argc, char* argv[])
{




	int device_select;

	if (argc == 2)
	{
		device_select = std::stoi(argv[1]);

	}
	else
	{
			cout << "--------" << endl;
			cout << "GPU List" << endl;
			cout << "--------" << endl;

			int devicesCount;
			cudaGetDeviceCount(&devicesCount);
			for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
			{
				cudaDeviceProp deviceProperties;
				cudaGetDeviceProperties(&deviceProperties, deviceIndex);
				cout << '[' << deviceIndex << ']' << ' ' << deviceProperties.name << endl;
			}

			cout << "Select GPU: ";

			cin >> device_select;

			cout << endl;
	}



	cudaDeviceProp deviceProperties;
	cudaGetDeviceProperties(&deviceProperties, device_select);
	cout << "Device " << '[' << device_select << ']' << ' ' << deviceProperties.name << " selected." << endl;


	cudaSetDevice(device_select);


	// Allow switching between float and double precision.
	// Note: May remove in future versions and assume double precision.
	typedef double PRECISION;

	//select model ID and initial condition ID
	// see solver_template.h for list of values

	bool is_fixed_newton_error = false;

//#define CONSTANT_ID
//
//#ifdef CONSTANT_ID
//	model::id const MODEL_ID = model::CONSANT;
//#endif
//
//#ifdef POLYMER_ID
//		model::id const MODEL_ID = model::POLYMER;
//#endif
//
//
//#ifdef NLC_ID
//	model::id const MODEL_ID = model::NLC;
//#endif
//
//

	model::id const MODEL_ID = model::NLC_ANCHORING;



	initial_condition::id const IC_ID = initial_condition::LOAD_FROM_FILE;

	// select boundary conditions
	// NOTE: For now only symmetric boundary conditions are implemented 
	//       i.e. h_x=h_xxx=0. Will upload revision to the code in the following
	//       months that allow a cleaner implementation of multiple boundary condition.
	//       You may alter boundary_conditions.h and implement your own boundary conditions
	boundary_condtion_type::IDs const BC_X0 = boundary_condtion_type::SYMMETRIC;
	boundary_condtion_type::IDs const BC_Y0 = boundary_condtion_type::SYMMETRIC;
	boundary_condtion_type::IDs const BC_XN = boundary_condtion_type::SYMMETRIC;
	boundary_condtion_type::IDs const BC_YM = boundary_condtion_type::SYMMETRIC;


	//boundary_condtion_postition::IDs const POS_ID = boundary_condtion_postition::SECOND;

	//PRECISION const test1 = boundary_condition_handler_gpu::get_stencil_coefficient_h_sss<PRECISION, BC_X0, BC_XN, POS_ID>(-2, 0);
	//PRECISION const test2 = boundary_condition_handler_gpu::get_stencil_coefficient_h_sss<PRECISION, BC_X0, BC_XN, POS_ID>(-1, 0);
	//PRECISION const test3 = boundary_condition_handler_gpu::get_stencil_coefficient_h_sss<PRECISION, BC_X0, BC_XN, POS_ID>( 0, 0);
	//PRECISION const test4 = boundary_condition_handler_gpu::get_stencil_coefficient_h_sss<PRECISION, BC_X0, BC_XN, POS_ID>( 1, 0);


	//PRECISION const test5 = boundary_condition_handler_gpu::get_stencil_coefficient_h_sss<PRECISION, BC_X0, BC_XN, POS_ID>(-2, 1);
	//PRECISION const test6 = boundary_condition_handler_gpu::get_stencil_coefficient_h_sss<PRECISION, BC_X0, BC_XN, POS_ID>(-1, 1);
	//PRECISION const test7 = boundary_condition_handler_gpu::get_stencil_coefficient_h_sss<PRECISION, BC_X0, BC_XN, POS_ID>(0, 1);
	//PRECISION const test8 = boundary_condition_handler_gpu::get_stencil_coefficient_h_sss<PRECISION, BC_X0, BC_XN, POS_ID>(1,1);

	double test=6;
	// simplifying class reference
	typedef gadit_solver<PRECISION, MODEL_ID, IC_ID, BC_X0, BC_XN, BC_Y0, BC_YM> gadit_solver;

	// File contain all parameters that can be altered by the user. 
	parameters<PRECISION, MODEL_ID, IC_ID> paras;

	// Spatial partition parameters
	paras.spatial.ds = 0.05;
	paras.spatial.x0 = 0.0;
	paras.spatial.n = 2000;
	paras.spatial.y0 = 0.0;
	paras.spatial.m = 2000;

	paras.spatial.ds = 0.02;
	paras.spatial.x0 = 0.0;
	paras.spatial.n = 1200;
	paras.spatial.y0 = 0.0;
	paras.spatial.m = 1200;

	paras.spatial.ds = 0.02;
	paras.spatial.x0 = 0.0;
	paras.spatial.n = 2074;
	paras.spatial.y0 = 0.0;
	paras.spatial.m = 2074;

	paras.spatial.ds = 0.05;
	paras.spatial.x0 = 0.0;
	paras.spatial.n = 2*1967;
	paras.spatial.y0 = 0.0;
	paras.spatial.m = 1967;

	// Model Parameters
	paras.model.cC = 0.0857;
	paras.model.cN = 1.67;
	paras.model.cK = 36.0;
	paras.model.b = 0.01;
	paras.model.beta = 1.0;
	paras.model.w = 0.05;
	paras.model.lambda = 0.75;
	paras.model.nu = 1.0 - paras.model.lambda;


	//paras.model.cC = 3.4875*pow(10,-5);
	//paras.model.ci = 0.70875;
	//paras.model.ASIO = 0.65651414;
	//paras.model.ASI = -3.8794017;

	//paras.model.d = 191;

	/*paras.model.c1 = 1.0;
	paras.model.c2 = 1.0;
*/

//	paras.model.lambda = 0.75;
//	paras.model.nu = 1- paras.model.lambda;

	//paras.model.Gamma = 1;
	//paras.model.K = 1;
	//paras.model.b = 10;
	//paras.model.c = 0.45;
	//paras.model.np = 0.5;
	//paras.model.phi_star = 0.64;
	 
	paras.model.compute_derived_parameters();

	double test123 = paras.model.getMaxGrowthMode(0.1);
	test123 = paras.model.getMaxGrowthMode(0.2);
	test123 = paras.model.getMaxGrowthMode(0.3);
	test123 = paras.model.getMaxGrowthMode(0.4);
	test123 = paras.model.getMaxGrowthMode(0.5);
	test123 = paras.model.getMaxGrowthMode(0.6);
	test123 = paras.model.getMaxGrowthMode(0.7);
	double test234 = test123;

	//paras.model.cC = 9.6875e-2;
	//paras.model.ci = 1.9687500000e-07;
	//paras.model.ASIO = 1.8236503896e-01;
	//paras.model.ASI = -1.0776115939e-00;
	//paras.model.d = 19.2;

	//paras.model.cC = 0.005812500000;
	//paras.model.ci = 1.181250000000;
	//paras.model.ASIO = 1.094190233757;
	//paras.model.ASI = -6.465669563108;
	//paras.model.d = 191;

	//paras.model.cC = 9.6875e-3;
	//paras.model.ci = 1.96875;
	//paras.model.ASIO = 1.4920775915;
	//paras.model.ASI = -18.236503896;
	//paras.model.d = 192;

	// Parameters for initial condition
	//paras.initial.h0 = 0.5;
	//paras.initial.epx = 0.01;
	//paras.initial.nx = 2 * 10;
	//paras.initial.epy = 0.01;
	//paras.initial.ny = 2 * 10;

	// Temporal parameters
	paras.temporal.t_start = 0;
	paras.temporal.dt_out = 1.5;
	paras.temporal.t_end    = paras.temporal.t_start  + paras.temporal.dt_out * 600;

	// backup time for solution in minutes 
	paras.backup.updateTime = 10;

	// Add '/' to end if not using execution directory as root e.g. some_folder/
	paras.io.root_directory = "";

	// Toggle to control output of status of GADIT solver
	paras.io.is_console_output = true;
	paras.io.is_full_text_output = false;


	// It is not necessary the change the remaining parameters,
	// but feel free to do so.
	paras.newton.error_tolerence = pow(10, -12);

	// Testing shows 10 produces best effective time step
	// i.e. dt/interation_count
	paras.newton.max_iterations = 10;
	// Applies a minimum amount iterations with out convergence checks
	paras.newton.min_iterations = 3;

	paras.temporal.dt_min = pow(10, -10);
	paras.temporal.dt_max = 0.1*paras.temporal.dt_out;

	// set large to prevent excessive dt increase
	// that with results in immediate newton convergence
	// failure within thresholds.
	paras.temporal.min_stable_step = 500;
	// dt is allowed to increase exponentially once min_step is
	// reach. After failure,  min_stable_steps much be achieved
	// before dt can be increased again.
	paras.temporal.dt_ratio_increase = 1.07;
	paras.temporal.dt_ratio_decrease = 1.05;

	// setting to a very small will only affect the
	// start up of GADIT. GADIT allows exponential growth
	// of the time step
	paras.temporal.dt_init = 0.00000001*paras.temporal.dt_out;

	// initializes solver and evolve solution
	gadit_solver *solver;


	solver = new gadit_solver();
	solver->initialize(paras);
	solver->solve_model_evolution();

//
//	// Convergence Study 
//
//	// Spatial
//
//	double h0 = 0.5;
//
//	
//	//
//	//
//
//#ifdef CONSTANT_ID
//
//	paras.model.c1 = 1.0;
//	paras.model.c2 = 1.0; 
//	h0 = 1;
//
//#endif
//
//#ifdef POLYMER_ID
//	paras.model.cC = 0.005812500000;
//	paras.model.ci = 1.181250000000;
//	paras.model.ASIO = 1.094190233757;
//	paras.model.ASI = -6.465669563108;
//
//	paras.model.d = 191;
//
//	//paras.model.cC = 0.000096875000;
//	//	paras.model.cC = 0.005812500000;
//	//paras.model.ci = 0.019687500000;
//	//	paras.model.ASIO = 0.018236503896;
//	//paras.model.ASI = -0.107761159385;
//
//
//	h0 = 3.9;
//#endif
//
//#ifdef NLC_ID
//	paras.model.cC = 0.0857;
//	paras.model.cN = 1.67;
//	paras.model.cK = 36.0;
//	paras.model.b = 0.01;
//	paras.model.beta = 1.0;
//	paras.model.w = 0.05;
//	h0 = 0.5;
//#endif
//
//
//	paras.model.compute_derived_parameters();
//	
//
//
//
//	paras.initial.epx = 0.1;
//	paras.initial.nx = 1;
//	paras.initial.epy = 0.1;
//	paras.initial.ny = 1;
//
//	paras.newton.max_iterations = 10;
//	paras.newton.error_tolerence = pow(10, -14);
//
//	paras.temporal.dt_min = pow(10, -14);
//
//	paras.temporal.dt_ratio_increase = 1.07;
//	paras.temporal.dt_ratio_decrease = 1.05;
//
//	paras.initial.h0 = h0;
//
//	int m = 12;
//	int n = 32;
//	double scale = 1.6;
//
//	double dscale = scale / (1.0*m);
//	double qm = paras.model.getMaxGrowthMode(h0);
//
//	printf("qm: %e\n", qm);
//
//	int sig_fig = 2;
//	double scale2 = pow(10.0, -ceil(log10(fabs(qm))) + 1 + sig_fig);
//
//	//qm = round(qm * scale2) / scale2;
//
//	printf("qm: %e\n", qm);
//
//	for (int i = 0; i < m; i++)
//	{
//
//		double q = qm *(i + 1)*dscale;
//		double w = paras.model.getGrowthRate(h0, q);
//
//		double dtout = 0.182321556793955 / abs(w);
//		//dtout = 0.693147180559945 / abs(w);
//		//dtout = 0.095310179804325 / abs(w);
//	//	dtout = 0.405465108108164 / abs(w);
//		dtout = 0.262364264467491 / abs(w);
//		scale2 = pow(10.0, -ceil(log10(fabs(dtout))) + 1 + sig_fig);
//
//		dtout = round(dtout * scale2) / scale2;
//
//
//		printf("q: %e\n", q);
//		printf("w: %e\n", w);
//		printf("t_end: %e\n", dtout);
//
//
//		double l =   PI / q;
//
//		scale2 = pow(10.0, -ceil(log10(fabs(l))) + 1 + sig_fig);
//
//		l = round(l * scale2) / scale2;
//
//
//		double ds = l / n;
//
//		if (i % 2 == 0)
//		{
//			paras.initial.epx = 0.1;
//			paras.initial.epy = 0.0;
//		}
//		else
//		{
//			paras.initial.epx = 0.0;
//			paras.initial.epy = 0.1;
//		}
//
//		paras.spatial.ds = ds;
//		paras.spatial.n = n;
//		paras.spatial.m = n;
//
//		paras.temporal.t_start = 0;
//
//
//
//		paras.temporal.dt_out = dtout/100;
//		paras.temporal.t_end =  dtout;
//
//		paras.temporal.dt_init = pow(10, -12);
//		paras.temporal.dt_max = 0.01*paras.temporal.dt_out;
//
//		paras.temporal.dt_min =pow(10,-13);
//
//		char buff[100];
//
//		sprintf(buff, "%i/", i);
//
//		paras.io.root_directory = buff;
//
//		solver = new gadit_solver();
//		solver->initialize(paras);
//		solver->solve_model_evolution();
//		free(solver);
//	}
//
//	

	//paras.newton.max_iterations = 10000;
	//

	//int N0 = 36;

	//int n_waves = 6;

	//paras.initial.epx = 0.1;
	//paras.initial.nx = n_waves;
	//paras.initial.epy = 0.1;
	//paras.initial.ny = n_waves;


	//paras.initial.h0 = h0;
	//double qm = paras.model.getMaxGrowthMode(h0);
	//double wm = paras.model.getGrowthRate(h0, qm);
	//double L = n_waves *2 * PI / qm;



	//int M = 6;
	//int P = 3;

	//double ds0 = L / N0;
	//double factor = 1;

	//double dt0 = factor*ds0*ds0*pow(P, -2 * (M - 1));
	//dt0 = pow(10, -12);
	//double dTout = 0.0000025 / wm;
	//dTout = 20 * 0.00025 / wm;
	//dTout = 0.000005 / wm;



	//int N_out_total = 20;

	//int N_out = dTout / dt0;

	//dt0 = pow(10,-8) / wm;

	//N_out = 1000;
	//dTout = dt0*N_out;
	//double Tend = N_out_total*dTout;

	////for (int i = M-1; i >= 0  ; i--)
	//for (int i = 0; i < M; i++)
	//{

	//	int N = N0 *pow(P, i);

	//	double dt = dt0;
	//	double ds = L / (1.0*N);


	//	paras.newton.error_tolerence = max(min(ds*ds, dt*dt),pow(10,-14));
	//	paras.newton.error_tolerence = max(max(ds*ds, ds*ds), pow(10, -14));

	//	if (is_fixed_newton_error)
	//		paras.newton.error_tolerence = pow(10, -14);
	//	else
	//		paras.newton.error_tolerence = max(max(ds*ds, ds*ds), pow(10, -14));
	//	paras.spatial.ds = ds;
	//	paras.spatial.n  = N;
	//	paras.spatial.m  = N;

	//	paras.temporal.t_start = 0;


	////	paras.temporal.dt_min = dt;
	//	paras.temporal.dt_max = ds;


	//	paras.temporal.dt_out  = 100*dt;
	//	paras.temporal.t_end = 1000*dt;
	////	paras.temporal.dt_out = dt;
	////	paras.temporal.t_end = dt;
	//	paras.temporal.dt_init = dt;


	//	char buff[100];

	//	sprintf(buff, "spatial/%i/", i);
	//
	//	paras.io.root_directory = buff;


	//	solver = new gadit_solver();
	//	solver->initialize(paras);
	//	solver->solve_model_evolution();
	//	free(solver);

	//}

	//paras.temporal.dt_ratio_increase = 1.00;
	//paras.temporal.dt_ratio_decrease = 2.00;

	//N0 = 256;

	//ds0 = L / (1.0*N0);
	//dt0 = factor*ds0*ds0;

	//M = 6;
	//dt0 = pow(10, -8);

	// N_out = dTout / dt0;

	// if (N_out < 1)
	// {
	//	 N_out = 1;
	// }

	// dTout = dt0*N_out;

	// dt0 = pow(10, -9) / wm;

	// N_out = 1000;
	// dTout = dt0*N_out;
	//  Tend = N_out_total*dTout;


	// Tend = N_out_total*dTout;

	// dt0 =  pow(10, -3) / wm;

	//for (int i = 0; i < M; i++)
	//{

	//	int N = N0;

	//	double dt = dt0*pow(2,-i);
	//	double ds = L / (1.0*N);


	//	paras.newton.error_tolerence = max(min(ds*ds, dt*dt), pow(10, -14));


	//	if (is_fixed_newton_error)
	//		paras.newton.error_tolerence = pow(10, -14);
	//	else
	//		paras.newton.error_tolerence = max(max(dt*dt, dt*dt), pow(10, -14));

	//

	//	paras.spatial.ds = ds;
	//	paras.spatial.n = N;
	//	paras.spatial.m = N;

	//	paras.temporal.t_start = 0;


	//	paras.temporal.dt_out = 100*dt0;
	//	paras.temporal.t_end = 1000*dt0;

	//	paras.temporal.dt_init = dt;

	//	paras.temporal.dt_min =dt;

	//	char buff[100];

	//	sprintf(buff, "temporal/%i/", i);

	//	paras.io.root_directory = buff;


	//	solver = new gadit_solver();
	//	solver->initialize(paras);
	//	solver->solve_model_evolution();
	//	free(solver);

	//}




	return 0;
};

