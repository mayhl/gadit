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
// Name:			initial_condition_list.h
// Version: 		1.0
// Purpose:			Example of how to execute solver
// ----------------------------------------------------------------------------------
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gadit_solver.h"
#include <stdio.h>


#include "output.h"
int main()
{


	// Allow switching between float and double precision.
	// Note: May remove in future versions and assume double precision.
	typedef double PRECISION;

	model::id const MODEL_ID = model::NLC;
	initial_condition::id const IC_ID = initial_condition::LINEAR_WAVES;
	boundary_condtion_type const BC_X0 = boundary_condtion_type::SYMMETRIC;
	boundary_condtion_type const BC_Y0 = boundary_condtion_type::SYMMETRIC;
	boundary_condtion_type const BC_XN = boundary_condtion_type::SYMMETRIC;
	boundary_condtion_type const BC_YM = boundary_condtion_type::SYMMETRIC;

	// simplifying class reference
	typedef gadit_solver<PRECISION, MODEL_ID, IC_ID, BC_X0, BC_XN, BC_Y0, BC_YM> gadit_solver;

	parameters<PRECISION, MODEL_ID, IC_ID> paras;


	int n = 519;
	int m = n;

	PRECISION ds = 0.02;

	PRECISION  x0 = 0.0;
	PRECISION  xn = n*ds;
	PRECISION  y0 = 0.0;
	PRECISION  ym = m*ds;

	// Must set all parameters 

	paras.initial.h0 = 0.3;
	paras.initial.epx = 0.01;
	paras.initial.nx = 2 * 1;
	paras.initial.epy = 0.01;
	paras.initial.ny = 2 * 1;

	paras.temporal.t_start = 0.0;
	paras.temporal.dt_out	= 2.0;
	paras.temporal.t_end = paras.temporal.dt_out * 500;

	paras.model.cC = 0.0857;
	paras.model.cN = 1.67;
	paras.model.cK = 36.0;
	paras.model.b = 0.01;
	paras.model.beta = 1.0;
	paras.model.w = 0.05;


	// Precompute complex expression  on CPU that are known at compile time.
	// Removes redundant calculations on GPU level.
	paras.model.compute_derived_parameters();

	paras.spatial.ds = 0.02;
	PRECISION qm = paras.model.getMaxGrowthMode(paras.initial.h0);
	PRECISION lambdam = (2 * PI / qm)/sqrt(1.0);
	ds = paras.spatial.ds;
	n = ceil(lambdam / ds);
	m = n;
	xn = ds*n;
	ym = ds*m;

	paras.spatial.x0 = 0.0;
	paras.spatial.n = n;

	paras.spatial.y0 = 0.0;
	paras.spatial.m = m;


	// Add '/' to if not using exeuction directory as root e.g. sim/ 
	paras.io.root_directory = "";
	paras.io.is_console_output = true;
	paras.io.is_full_text_output = false;

	// Tuning Parameters, may leave. 

	paras.newton.error_tolerence = pow(10, -10);

	// Testing shows 10 produces best effective timestep
	// i.e. dt/interation_count
	paras.newton.max_iterations = 10;
	// Applies minimum iterations with out convergence checks
	paras.newton.min_iterations = 3;

	paras.temporal.dt_min = pow(10, -13);
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

	paras.temporal.dt_init = 0.0001*paras.temporal.dt_out;

	// backup time in minutes 
	paras.backup.updateTime = 5;


	std::string test = paras.to_string();
	printf(test.c_str());
	std::string test1 = paras.spatial.to_string();


	// initializes solver and evolve solution
	//gadit_solver *solver;
	//solver = new gadit_solver();

	//solver->initialize(paras);
	//solver->solve_model_evolution();

	//typedef double DATATYPE;
	//	timestep_manager<DATATYPE> t_mang;
	//	temporal_parameters<DATATYPE> parameters ;


	//	parameters.t_start = 0.0;
	//	parameters.dt_init = 0.3;
	//	parameters.dt_out  = 1.0;
	//	parameters.min_stable_step	= 3;
	//	parameters.dt_ratio_increase	= 1.07;
	//	parameters.dt_ratio_decrease	= 1.05;
	//	
	//parameters.dt_min				= pow(10,-13);
	//parameters.dt_max				= 0.1*parameters.dt_out;

	//	t_mang.initialize( parameters );
	//	
	//	newton_status::status test;

	//	for ( int i = 0 ; i < 10 ; i++ )
	//	{
	//	test = newton_status::SUCCESS;
	//	t_mang.update_dt( test );
	//	}
    
	//// Allow switching between float and double precision.
	//// Note: May remove in future versions and assume double precision.
	//typedef double PRECISION;
	//
	//model::id const MODEL_ID = model::NLC;
	//initial_condition::id const IC_ID = initial_condition::LINEAR_WAVES;
	//boundary_condtion_type const BC_X0 = boundary_condtion_type::SYMMETRIC;
	//boundary_condtion_type const BC_Y0 = boundary_condtion_type::SYMMETRIC;
	//boundary_condtion_type const BC_XN = boundary_condtion_type::SYMMETRIC;
	//boundary_condtion_type const BC_YM = boundary_condtion_type::SYMMETRIC;
	//
	//// simplifying class reference
	//typedef gadit_solver<PRECISION, model::NLC, IC_ID, BC_X0, BC_XN, BC_Y0, BC_YM> nlc_solver;
	//parameters<PRECISION, model::NLC, IC_ID> nlc_paras;

	//typedef gadit_solver<PRECISION, model::POLYMER, IC_ID, BC_X0, BC_XN, BC_Y0, BC_YM> polymer_solver;
	//parameters<PRECISION, model::POLYMER, IC_ID> polymer_paras;

	//typedef gadit_solver<PRECISION, model::CONSANT, IC_ID, BC_X0, BC_XN, BC_Y0, BC_YM> constant_solver;
	//parameters<PRECISION, model::CONSANT, IC_ID> constant_paras;


	//io_parameters paras_io;
	//newton_parameters<PRECISION> paras_newton;
	//temporal_parameters<PRECISION> paras_temporal;
	//spatial_parameters<PRECISION> spatial_temporal;
	//backup_parameters paras_backup;
	//initial_parameters<PRECISION,IC_ID> paras_initial;

	//paras_io.root_directory = "";
	//paras_io.is_console_output = true;
	//paras_io.is_full_text_output = false;

	//paras_newton.error_tolerence = pow(10, -10);
	//paras_newton.max_iterations = 10;
	//paras_newton.min_iterations = 3;

	//paras_temporal.dt_min = pow(10, -13);
	//paras_temporal.dt_max = pow(10, -12);
	//paras_temporal.min_stable_step = 500;
	//paras_temporal.dt_ratio_increase = 1.07;
	//paras_temporal.dt_ratio_decrease = 1.05;
	//paras_temporal.dt_init = pow(10, -12);;

	//paras_backup.updateTime = 5;

	//paras_initial.epx = 0.01;
	//paras_initial.nx = 2 * 1;
	//paras_initial.epy = 0.01;
	//paras_initial.ny = 2 * 1;


	//nlc_paras.model.cC = 0.0857;
	//nlc_paras.model.cN = 1.67;
	//nlc_paras.model.cK = 36.0;
	//nlc_paras.model.b = 0.01;
	//nlc_paras.model.beta = 1.0;
	//nlc_paras.model.w = 0.05;
	//nlc_paras.model.compute_derived_parameters();

	//nlc_paras.io = paras_io;
	//nlc_paras.newton = paras_newton;
	//nlc_paras.temporal = paras_temporal;
	//nlc_paras.backup = paras_backup;
	//nlc_paras.initial = paras_initial;
	//nlc_paras.initial.h0 = 0.2;


	//polymer_paras.model.cC = 9.6875e-3;
	//polymer_paras.model.ci = 1.96875;
	//polymer_paras.model.ASIO = 1.9894367886;
	//polymer_paras.model.ASI = -10.776115939;
	//polymer_paras.model.d = 192;

	//polymer_paras.io = paras_io;
	//polymer_paras.newton = paras_newton;
	//polymer_paras.temporal = paras_temporal;
	//polymer_paras.backup = paras_backup;
	//polymer_paras.initial = paras_initial;
	//polymer_paras.initial.h0 = 3.9;


	//constant_paras.model.c1 = 1.0;
	//constant_paras.model.c2 = 1.0;

	//constant_paras.io = paras_io;
	//constant_paras.newton = paras_newton;
	//constant_paras.temporal = paras_temporal;
	//constant_paras.backup = paras_backup;
	//constant_paras.initial = paras_initial;
	//constant_paras.initial.h0 = 1.0;


	//int n = 519;
	//int m = n;
	//

	//PRECISION  x0 = 0.0;
	//PRECISION  xn = n*ds;
	//PRECISION  y0 = 0.0;
	//PRECISION  ym = m*ds;

	//// Must set all parameters 




	//PRECISION qm = nlc_paras.model.getMaxGrowthMode(nlc_paras.initial.h0 );


	//size_t const q_count = 9;

	//PRECISION qm_ratios[q_count] = { 0.2 , 0.4 , 0.6 , 0.8 , 1.0 , 1.2 , 1.4 , 1.6 , 1.8};

	//PRECISION lambda = 2*PI/qm;
	//PRECISION g_rate = nlc_paras.model.getGrowthRate(nlc_paras.initial.h0, qm);


	//PRECISION ds = 0.01;
	//n = 100;
	//m = 100;


	//n = ceil( lambda/ds );
	//m=n;
	//xn=ds*n;
	//ym=ds*m;


	//nlc_solver *n_solver;
	//n_solver = new nlc_solver();

	//n_solver->initialize(nlc_paras);
	//n_solver->solve_model_evolution();

    return 0;
}

