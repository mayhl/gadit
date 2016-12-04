#pragma once
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
	typedef gadit_solver<PRECISION, MODEL_ID, IC_ID , BC_X0, BC_XN, BC_Y0, BC_YM> gadit_solver;
	
	parameters<PRECISION,MODEL_ID,IC_ID> paras;

	
	int n = 519;
	int m = n;
	
	PRECISION ds = 0.01;

	PRECISION  x0 = 0.0;
	PRECISION  xn = n*ds;
	PRECISION  y0 = 0.0;
	PRECISION  ym = m*ds;

	// Must set all parameters 
	
	paras.initial.h0					= 0.24;
	paras.initial.epx					= 0.01;
	paras.initial.nx					= 2*1;
	paras.initial.epy					= 0.01;
	paras.initial.ny					= 2*1;

	paras.temporal.t_start				= 0.0;
	paras.temporal.dt_out		
		= 1.0;
	paras.temporal.t_end				= paras.temporal.dt_out*500;
	
	paras.model.cC						= 0.0857;
	paras.model.cN						= 1.67;
	paras.model.cK						= 36.0;
	paras.model.b						= 0.01;
	paras.model.beta					= 1.0;
	paras.model.w   					= 0.05;
	
	
	// Precompute complex expression  on CPU that are known at compile time.
	// Removes redundant calculations on GPU level.
	paras.model.compute_derived_parameters();

	PRECISION qm = paras.model.getMaxGrowthMode( paras.initial.h0 );
	PRECISION lambdam = 2*PI/qm;

	n = ceil( lambdam/ds );
	m=n;
	xn=ds*n;
	ym=ds*m;

	paras.spatial.x0					= 0.0;
	paras.spatial.xn					= xn;
	paras.spatial.n						= n;

	paras.spatial.y0					= 0.0;
	paras.spatial.ym					= ym;
	paras.spatial.m						= m;


	// Add '/' to if not using exeuction directory as root e.g. sim/ 
	paras.io.root_directory			= "";
	paras.io.is_console_output		= true;
	paras.io.is_full_text_output		= false;
	
	// Tuning Parameters, may leave. 
	
	paras.newton.error_tolerence		= pow(10, -10);
	
	// Testing shows 10 produces best effective timestep
	// i.e. dt/interation_count
	paras.newton.max_iterations		= 10;
	// Applies minimum iterations with out convergence checks
	paras.newton.min_iterations		= 3;

	paras.temporal.dt_min			= pow(10,-13);
	paras.temporal.dt_max			= 0.1*paras.temporal.dt_out;
	
	// set large to prevent excessive dt increase
	// that with results in immediate newton convergence
	// failure within thresholds.
	paras.temporal.min_stable_step		= 500;
	// dt is allowed to increase exponentially once min_step is
	// reach. After failure,  min_stable_steps much be achieved
	// before dt can be increased again.
	paras.temporal.dt_ratio_increase	= 1.07;
	paras.temporal.dt_ratio_decrease	= 1.05;

	paras.temporal.dt_init			= 0.0001*paras.temporal.dt_out;

	// backup time in minutes 
	paras.backup.updateTime				= 5;

    // initializes solver and evolve solution
	gadit_solver *solver;
	solver = new gadit_solver();

	solver->initialize(paras);
	solver->solve_model_evolution();




    return 0;
}

