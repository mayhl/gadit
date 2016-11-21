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

template <typename DATATYPE, model_id MODEL_ID , 
	boundary_condtion_type BC_X0, boundary_condtion_type BC_XN,
	boundary_condtion_type BC_Y0, boundary_condtion_type BC_YM> class gadit_solver
{
public:




	dimensions dims;
	parameters<DATATYPE,MODEL_ID> paras;
	unified_work_space<DATATYPE> u_ws;
	cuda_parameters::kernal_launch_parameters ker_launch_paras;
	
	gadit_solver(){};

	void initialize(parameters<DATATYPE,MODEL_ID> paras ,  inital_condition_list::ic_id ic )
	{
		this->paras = paras;
		setup_partition_and_initialize_memory( paras.spatial.x0 , paras.spatial.xn , paras.spatial.n ,
												paras.spatial.y0 , paras.spatial.ym , paras.spatial.m );
		set_and_load_initial_condition(ic);
	}
	

	void solve_model_evolution()
	{

		string outputString;
		timestep_manager<DATATYPE> t_mang;
		backup_manager<DATATYPE> b_mang;
		status_logger<DATATYPE> log_file;
		
		newton_status::status n_status;
		timestep_manager_status::status t_status;

		b_mang.initialize( paras.backup);


		bool isFirstRun = !fileExists( file_directories::backupFileInfo );

		// loading data if temporary data exisits.
		if ( isFirstRun )
		{
			file_directories::clean();

			outputString =  get_time_stamp() + "Started simulations from initial condition.";
			write_to_new_file( file_directories::statusData , outputString );
			
			t_mang.initialize( paras.temporal );
		}
		else
		{		
			load_object<timestep_manager<DATATYPE>>(file_directories::backupFileInfo , t_mang );
			
			char buff[100];				
			sprintf(buff,"Continuing simulations from backup data at t = %11.10E." , t_mang.get_current_time() );
			outputString =  get_time_stamp() + buff;
			write_to_old_file( file_directories::statusData , outputString );
		}

		// loops till t_end reached or break statement
		// is executed by a failure state.
		while( t_mang.is_not_completed() )
		{	
			size_t newton_count;
			DATATYPE dt;
			DATATYPE t;
			size_t	 timestep;
			
			newton_iterative_method::solve_time_step<DATATYPE, MODEL_ID, BC_X0, BC_XN, BC_Y0, BC_YM>(u_ws, dims, paras ,  ker_launch_paras , dt , newton_count , n_status );

			//char buff2[100];			
			//sprintf(buff2,"dt = %11.10E , t = %11.10E" , t_mang.get_timestep() , t_mang.get_current_time() );	
			//outputString =  get_time_stamp() + buff2;
			//write_to_old_file( file_directories::statusData , outputString );

			////bool is_timestep_successful = (n_status == newton_status::SUCCESS);
			//bool is_not_handled_state = false;


			//
			//if ( n_status != newton_status::SUCCESS )
			//{
			//	switch( n_status )
			//	{
			//	case newton_status::INCREASE_DT:
			//		// Value should not be returned by newton solver. Dummy case to remove from default case.
			//		// See 'newton_status' enum for further details.
			//		break;
			//	case newton_status::CONVERGENCE_FAILURE_LARGE:
			//			char buff[100];			
			//		
			//			sprintf(buff,"Newton Failed, dt = %11.10E , t = %11.10E." , t_mang.get_timestep() , t_mang.get_current_time() );	
			//			outputString =  get_time_stamp() + buff;
			//			write_to_old_file( file_directories::statusData , outputString );
			//		break;
			//
			//
			//	default:
			//		is_not_handled_state = true;
			//		break;
			//	}
			//	
			//}
			//
			//if ( is_not_handled_state )
			//{
			//	outputString =  get_time_stamp() + "Simulation Failed! Invalid newton_status value returned.";
			//	write_to_old_file( file_directories::statusData , outputString );
			//	break;
			//}
			
			
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
				write_to_old_file( file_directories::statusData , outputString );
				break;
			}


			if ( t_mang.is_sucessful_output_step() )
			{
				char buff[100];		
				
				sprintf(buff,"Saving solution at t = %11.10E to file." , t_mang.get_current_time() );
				outputString =  get_time_stamp() + buff;
				write_to_old_file( file_directories::statusData , outputString );

				sprintf(buff, "/solution_%07d.dat", t_mang.get_next_output_index() );
				outputString = file_directories::outputDir + buff;
				memory_manager::copyDeviceToHost<DATATYPE>( u_ws.h );
				output_binary( outputString , u_ws.h->data_host , dims.n_pad , dims.m_pad );
			}

			if ( b_mang.is_backup_time() )
			{
				char buff[100];		
				
				sprintf(buff,"Backing up solution at t = %11.10E to file." , t_mang.get_current_time() );
				outputString =  get_time_stamp() + buff;
				write_to_old_file( file_directories::statusData , outputString );

				memory_manager::copyDeviceToHost<DATATYPE>( u_ws.h );
				output_binary( file_directories::backupSolution , u_ws.h->data_host , dims.n_pad , dims.m_pad );

				save_object<timestep_manager<DATATYPE>>(file_directories::backupFileInfo , t_mang );

				log_file.commit_data_to_files();
			}

		}

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

		paras.spatial.set_partition(ds, x_0, x_n, y_0, y_m);

		int padding = cuda_parameters::CELL_BORDER_PADDING;

		dims.set_dimensions(n, m, padding);

		int  initial_step_size = nonlinear_penta_solver::INITIAL_STEP_SIZE;
		int  down_solve_sub_loop_size = nonlinear_penta_solver::DOWN_SOLVE_SUB_LOOP;
		int  thread_size = cuda_parameters::PENTA_LU_LINE_THREAD_SIZE;

		dims.set_penta_dimension(initial_step_size, down_solve_sub_loop_size, thread_size);

		int reduction_block_size = cuda_parameters::SIMPLE_SQUARE_BLOCK;

		dims.set_reduction_dimension(reduction_block_size);

		int Jy_F_y_subloop_size = cuda_parameters::SOLVE_JY_SUBLOOP_SIZE;
		int Jy_F_y_thread_size = cuda_parameters::SOLVE_JY_THREAD_SIZE;

		dims.set_Jx_F_dimension(Jy_F_y_thread_size,Jy_F_y_subloop_size);

		int simple_block_size = cuda_parameters::SIMPLE_SQUARE_BLOCK;

		dims.set_simple_block_dimension(simple_block_size);

		u_ws.initalize_memory(dims);

		ker_launch_paras.initalize(dims);

		for (int i = 0; i < dims.n; i++)
		{
			u_ws.x->data_host[i] = x_0 + (i + 0.5)*ds;
		}
		for (int j = 0; j < dims.m; j++)
		{
			u_ws.y->data_host[j] =y_0 + (j + 0.5)*ds;
		}

	};

	void set_and_load_initial_condition(inital_condition_list::ic_id id )
	{
		inital_condition_list::compute_ic( u_ws.x , u_ws.y , u_ws.h , dims , paras.spatial , id );
		memory_manager::copyHostToDevice(u_ws.h);

	};

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


};





#endif

