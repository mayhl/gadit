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
// Name:			dimensions.h
// Version: 		1.0
// Purpose:		Contains dimensions of solution, padded solution, andi
//						relevant dimensions of grids of subdomains. 
// ----------------------------------------------------------------------------------

#ifndef DIMENSIONS
#define DIMENSIONS

struct dimensions
{
	int n;
	int m;

	int n_pad;
	int m_pad;

	int n_reduced;
	int m_reduced;
	
	int index_shift_x;
	int index_shift_y;

	int padding;

	int penta_lu_down_end_loop_size_x;
	int penta_lu_down_end_loop_size_y;

	int penta_lu_up_end_loop_size_x;
	int penta_lu_up_end_loop_size_y;


	int Jy_F_last_y_block_size;
	int Jy_F_last_x_block_size;

	int simple_last_block_x_size;
	int simple_last_block_y_size;

	dimensions() :n(-1), m(-1), n_pad(-1), m_pad(-1){};


	void set_dimensions(int n, int m, int padding)
	{

		this->n = n;
		this->m = m;
		this->padding = padding;
		this->n_pad = n + 2 * padding;
		this->m_pad = m + 2 * padding;
		this->index_shift_x = padding*n+padding;
		this->index_shift_y = padding*m+padding;

	}

	void set_penta_dimension(int initial_step_size, int down_solve_sub_loop, int  thread_size)
	{
		penta_lu_down_end_loop_size_x = (n - initial_step_size) % down_solve_sub_loop;
		penta_lu_down_end_loop_size_y = (m - initial_step_size) % down_solve_sub_loop;

		penta_lu_up_end_loop_size_x = n % thread_size;
		penta_lu_up_end_loop_size_y = m % thread_size;

		if (penta_lu_up_end_loop_size_x == 0) penta_lu_up_end_loop_size_x = thread_size;
		if (penta_lu_up_end_loop_size_y == 0) penta_lu_up_end_loop_size_y = thread_size;

	}
	void set_reduction_dimension(int block_size)
	{
		n_reduced = (n - 1 + block_size) / block_size;
		m_reduced = (m - 1 + block_size) / block_size;
	}

	void set_Jx_F_dimension(int thread_size, int block_size)
	{
		Jy_F_last_x_block_size = n % thread_size;
		Jy_F_last_y_block_size = m % block_size;

		if (Jy_F_last_x_block_size == 0) Jy_F_last_x_block_size = thread_size;
		if (Jy_F_last_y_block_size == 0) Jy_F_last_y_block_size = block_size;
	}

	void set_simple_block_dimension(int block_size)
	{
		simple_last_block_x_size = n % block_size;
		if (simple_last_block_x_size == 0) simple_last_block_x_size = block_size;

		simple_last_block_y_size = m % block_size;
		if (simple_last_block_y_size == 0) simple_last_block_y_size = block_size;
	}
};

#endif