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
// Name:		gadit_solver.h
// Version: 	1.0
// Purpose:		Main interface to solver. 
// Output:		Output solutions with ghost points.
// ----------------------------------------------------------------------------------

#ifndef GADIT_SOLVER
#define GADIT_SOLVER


#include "cuda_runtime.h"

#include "initial_condition_list.h"
#include "newton_iterative_method.h"
#include "backup_manager.h"
#include "timestep_manager.h"
#include "status_logger.h"


#include "output.h"

#include <chrono>
#include <ctime>

template <typename DATATYPE, model::id MODEL_ID , initial_condition::id IC_ID, 
	boundary_condtion_type BC_X0, boundary_condtion_type BC_XN,
	boundary_condtion_type BC_Y0, boundary_condtion_type BC_YM> class gadit_solver
{
public:

	gadit_solver(){};

	void initialize(parameters<DATATYPE,MODEL_ID,IC_ID> paras )
	{
		paras.spatial.compute_derived_parameters();
		paras.model.compute_derived_parameters();
		this->paras = paras;

		int padding = cuda_parameters::CELL_BORDER_PADDING;

		dims.set_dimensions(paras.spatial.n, paras.spatial.m, padding);

		int  initial_step_size = nonlinear_penta_solver::INITIAL_STEP_SIZE;
		int  down_solve_sub_loop_size = nonlinear_penta_solver::DOWN_SOLVE_SUB_LOOP;
		int  thread_size = cuda_parameters::PENTA_LU_LINE_THREAD_SIZE;

		dims.set_penta_dimension(initial_step_size, down_solve_sub_loop_size, thread_size);

		int reduction_block_size = cuda_parameters::SIMPLE_SQUARE_BLOCK;

		dims.set_reduction_dimension(reduction_block_size);

		int Jy_F_y_subloop_size = cuda_parameters::SOLVE_JY_SUBLOOP_SIZE;
		int Jy_F_y_thread_size = cuda_parameters::SOLVE_JY_THREAD_SIZE;

		dims.set_Jx_F_dimension(Jy_F_y_thread_size, Jy_F_y_subloop_size);

		int simple_block_size = cuda_parameters::SIMPLE_SQUARE_BLOCK;

		dims.set_simple_block_dimension(simple_block_size);

		u_ws.initalize_memory(dims);

		ker_launch_paras.initalize(dims);

		for (int i = 0; i < dims.n; i++)
		{
			u_ws.x->data_host[i] = paras.spatial.x0 + (i + 0.5)*paras.spatial.ds;
		}
		for (int j = 0; j < dims.m; j++)
		{
			u_ws.y->data_host[j] = paras.spatial.y0 + (j + 0.5)*paras.spatial.ds;
		}

	}


	void solve_model_evolution()
	{

		string outputString;
		timestep_manager<DATATYPE> t_mang;
		backup_manager<DATATYPE> sol_b_mang;
		status_logger<DATATYPE> log_file;
		
		newton_status::status n_status;
		timestep_manager_status::status t_status;

		sol_b_mang.initialize( paras.backup.updateTime );

		bool isFirstRun = !fileExists( file_directories::backupFileInfo );

		// loading data if temporary data exists.
		if ( isFirstRun )
		{
			file_directories::make_directories(paras.io.root_directory);
			file_directories::clean();
			write_to_new_file( paras.io.root_directory + file_directories::parameterData, paras.to_string() , false);

			t_mang.initialize( paras.temporal );
			initial_condition_list::compute<DATATYPE,IC_ID>( u_ws.x , u_ws.y , u_ws.h , dims , paras.spatial , paras.initial , paras.io);

			outputString =  get_time_stamp() + "Started simulations from initial condition.";
			write_to_new_file(paras.io.root_directory + file_directories::statusData , outputString, paras.io.is_console_output);
		}
		else
		{		

			load_object<timestep_manager<DATATYPE>>( paras.io.root_directory + file_directories::backupFileInfo , t_mang );
			load_binary ( file_directories::backupSolution , u_ws.h->data_host , dims.n_pad , dims.m_pad );
			

			//t_mang.t = 1930.0;
			//t_mang.outputStep = 193;
			//t_mang.t_next = t_mang.t + t_mang.parameters.dt_out;

			//t_mang.dt = 0.0001;
			//t_mang.t_next = t_mang.t + t_mang.dt;

			char buff[100];				
			sprintf(buff,"Continuing simulations from backup data at t = %11.10E." , t_mang.get_current_time() );
			outputString =  get_time_stamp() + buff;
			write_to_old_file( paras.io.root_directory + file_directories::statusData , outputString , paras.io.is_console_output );
		}

		memory_manager::copyHostToDevice(u_ws.h);

		// loops till t_end reached or break statement
		// is executed by a failure state.
		while( t_mang.is_not_completed() ) 
		{	
			size_t newton_count;
			
			newton_iterative_method::solve_time_step<DATATYPE, MODEL_ID,IC_ID, BC_X0, BC_XN, BC_Y0, BC_YM>(u_ws, dims, paras ,  ker_launch_paras , t_mang.get_timestep() , newton_count , n_status );

			if (paras.io.is_full_text_output) 
				output_all_timestep_changes<DATATYPE>(t_mang, n_status, paras.io.is_console_output);

			t_status = t_mang.update_dt( n_status );

			log_file.add_entry( t_mang.get_iteration_index() , t_mang.get_timestep() , n_status , newton_count );

			// Check adaptive time-stepping is working
			if ( t_status != timestep_manager_status::SUCCESS )
			{
				char buff[100];			

				switch( t_status )
				{
				case timestep_manager_status::MIN_DT:	
					sprintf(buff,"Simulation failed! Timestep below minimum threshold ,dt = %11.10E." , t_mang.get_timestep() );	
					break;
				case timestep_manager_status::DT_CHANGE_OUTPUT:	
					sprintf(buff,"Unexpected State! Newton iteration failed on lowering dt to match time output ,dt = %11.10E." , t_mang.get_timestep() );	
					break;
				default:
					sprintf(buff,"Unhandled timestep_manager_status." );	
					break;

				}
				outputString =  get_time_stamp() + buff;
				write_to_old_file( paras.io.root_directory + file_directories::statusData , outputString , paras.io.is_console_output);
				break;
			}

			if ( t_mang.is_sucessful_output_step() )
			{
				char buff[100];		
				
				sprintf(buff,"Saving solution at t = %11.10E to file." , t_mang.get_current_time() );
				outputString =  get_time_stamp() + buff;
				write_to_old_file( paras.io.root_directory +  file_directories::statusData , outputString, paras.io.is_console_output);

				sprintf(buff, "/solution_%07d.bin", t_mang.get_next_output_index() );

				std::string outputFileDir;
				outputFileDir = paras.io.root_directory + file_directories::outputDir + buff;
				memory_manager::copyDeviceToHost<DATATYPE>( u_ws.h );
				output_binary( outputFileDir , u_ws.h->data_host , dims.n_pad , dims.m_pad );
			}

			if ( sol_b_mang.is_backup_time() )
			{
				char buff[100];		
				
				sprintf(buff,"Backing up solution at t = %11.10E to file." , t_mang.get_current_time() );
				outputString =  get_time_stamp() + buff;
				write_to_old_file(  paras.io.root_directory + file_directories::statusData , outputString, paras.io.is_console_output);

				memory_manager::copyDeviceToHost<DATATYPE>( u_ws.h );
				output_binary(  paras.io.root_directory + file_directories::backupSolution , u_ws.h->data_host , dims.n_pad , dims.m_pad );

				save_object<timestep_manager<DATATYPE>>( paras.io.root_directory + file_directories::backupFileInfo , t_mang );

				log_file.commit_data_to_files(paras.io.root_directory);
			}

		}

		char buff[100];

		sprintf(buff, "Backing up solution at t = %11.10E to file.", t_mang.get_current_time());
		outputString = get_time_stamp() + buff;
		write_to_old_file(paras.io.root_directory + file_directories::statusData, outputString, paras.io.is_console_output);

		memory_manager::copyDeviceToHost<DATATYPE>(u_ws.h);
		output_binary(paras.io.root_directory + file_directories::backupSolution, u_ws.h->data_host, dims.n_pad, dims.m_pad);

		save_object<timestep_manager<DATATYPE>>(paras.io.root_directory + file_directories::backupFileInfo, t_mang);

		log_file.commit_data_to_files(paras.io.root_directory);

	};

	void clean_workspace()
	{
		u_ws.clean_workspace();
	}

private:
		
	void setup_partition_and_initialize_memory(DATATYPE x_0, DATATYPE x_n, int n,
		DATATYPE y_0, DATATYPE y_m, int m)
	{
		DATATYPE ds = (x_n - x_0) / (1.0*n);



	};

	template <typename DATATYPE> void output_all_timestep_changes(timestep_manager<DATATYPE> t_mang, newton_status::status n_status, bool  is_console_output)
	{

		string outputString;
		char buff[100];
		sprintf(buff, "dt = %11.10E , t = %11.10E", t_mang.get_timestep(), t_mang.get_current_time());
		outputString = get_time_stamp() + buff;
		write_to_old_file(file_directories::statusData, outputString, is_console_output);

		if (n_status != newton_status::SUCCESS)
		{
			switch (n_status)
			{
			case newton_status::INCREASE_DT:
				// Value should not be returned by newton solver. Dummy case to remove from default case.
				// See 'newton_status' enum for further details.
				break;
			case newton_status::CONVERGENCE_FAILURE_LARGE:
				sprintf(buff, "Newton Failed, dt = %11.10E , t = %11.10E.", t_mang.get_timestep(), t_mang.get_current_time());
				outputString = get_time_stamp() + buff;
				write_to_old_file(paras.io.root_directory + file_directories::statusData, outputString , paras.io.is_console_output );
				break;

			}

		}

	}
	string get_time_stamp()
	{
		string time_stamp;
		
		std::chrono::system_clock::time_point p = std::chrono::system_clock::now();
		std::time_t t = std::chrono::system_clock::to_time_t(p);
		time_stamp = std::ctime(&t);
		time_stamp.replace( time_stamp.end()-1 , time_stamp.end() , ":" );
		time_stamp += " ";
		return time_stamp;

	}
	
	dimensions dims;
	parameters<DATATYPE,MODEL_ID,IC_ID> paras;
	unified_work_space<DATATYPE> u_ws;
	cuda_parameters::kernal_launch_parameters ker_launch_paras;

};





#endif

