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

#ifndef MEMORY_MANAGER
#define MEMORY_MANAGER

#include <vector>
#include <cstdarg>
#include <iostream>
//#include "CudaHelperFunctions.h"
#include "cuda_runtime.h"



// -------------------------------------------------------------------- //
//                                                                      //
// Memory Manager V 1.0                                                 //
//                                                                      //
// Purpose: Wrapper functions to allocate/deallocate memory on host     //
//          and/or device when appropriate.                             //
//                                                                      //
// Notes: 1) Uses variadic function form, a recursive style calling     //
//           procedure for variable number of arguments.                //
//                                                                      //
//        2) Pinned memory leads to faster transfer times between host  //
//           and device, but should not be over used.                   //
//                                                                      //
// Requirements: At least C++ version 11 or Visual Studio 2013.         //
//                                                                      //
// -------------------------------------------------------------------- //
namespace memory_manager
{

	enum memory_scope{
		HOST_ONLY,
		DEVICE_ONLY,
		PINNED,
		NON_PINNED,
		//TEXTURE,
	};


	template<class DATATYPE>struct memory_unit{

	private:
		memory_scope scope;
		size_t memory_size;
		size_t n_x;
		size_t n_y;
		size_t n_z;
		unsigned short int dimensions;
		bool isDeviceClone;
		char *name;
		memory_unit<DATATYPE> *primary;

		void computeMemorySize()
		{
			memory_size = n_x*n_y*n_z*sizeof(DATATYPE);

			dimensions = 0;
			if (n_x > 1) { dimensions++; }
			if (n_y > 1) { dimensions++; }
			if (n_z > 1) { dimensions++; }
		}

	public:

		DATATYPE *data_host;
		DATATYPE *data_device;


		memory_scope getScope()
		{
			return scope;
		}

		size_t getMemorySize()
		{
			return memory_size;
		}

		size_t getSize_x()
		{
			return n_x;
		}
		size_t getSize_y()
		{
			return n_y;
		}
		size_t getSize_z()
		{
			return n_z;
		}

		memory_unit(memory_scope scope, int n_x, int n_y, int n_z) :
			scope(scope), n_x(n_x), n_y(n_y), n_z(n_z)
		{
			isDeviceClone = false;
			computeMemorySize();
		}
		memory_unit(memory_scope scope, int n_x, int n_y) :
			scope(scope), n_x(n_x), n_y(n_y), n_z(1)
		{
			isDeviceClone = false;
			computeMemorySize();
		}
		memory_unit(memory_scope scope, int n_x) :
			scope(scope), n_x(n_x), n_y(1), n_z(1)
		{
			isDeviceClone = false;
			computeMemorySize();
		}
		memory_unit(memory_unit *copy) :
			scope(copy->scope), n_x(copy->n_x), n_y(copy->n_y), n_z(copy->n_z), memory_size(copy->memory_size)
		{
			isDeviceClone = false;
		}

	};


	template<typename DATATYPE, typename ...Types>void freeAll(memory_unit<DATATYPE> *first, Types* ... rest)
	{
		freeAll(first);
		freeAll(rest...);
	}

	template<typename DATATYPE, typename ...Types>void freeAll(memory_unit<DATATYPE> *first)
	{
		switch (first->getScope())
		{
		case memory_scope::HOST_ONLY:
			free(first->data_host);
			break;

		case memory_scope::DEVICE_ONLY:
			cudaFree(first->data_device);
			break;

		case memory_scope::PINNED:
			cudaFreeHost(first->data_host);
			cudaFree(first->data_device);
			break;

		case memory_scope::NON_PINNED:
			free(first->data_host);
			cudaFree(first->data_device);
			break;

		}


	};

	template<typename DATATYPE, typename ...Types>void copyDeviceToHost(memory_unit<DATATYPE> *first, Types* ... rest)
	{
		copyDeviceToHost(first);
		copyDeviceToHost(rest...);
	}

	template<typename DATATYPE, typename ...Types>void copyDeviceToHost(memory_unit<DATATYPE> *first)
	{
		size_t datasize = first->getMemorySize();

		switch (first->getScope())
		{
		case memory_scope::HOST_ONLY:
			break;

		case memory_scope::DEVICE_ONLY:
			break;

		case memory_scope::PINNED:
			cudaMemcpy(first->data_host, first->data_device, datasize, cudaMemcpyDeviceToHost);
			break;

		case memory_scope::NON_PINNED:
			cudaMemcpy(first->data_host, first->data_device, datasize, cudaMemcpyDeviceToHost);
			break;
		}
	}


	template<typename DATATYPE, typename ...Types>void copyHostToDevice(memory_unit<DATATYPE> *first, Types* ... rest)
	{
		copyHostToDevice(first);
		copyHostToDevice(rest...);
	}

	template<typename DATATYPE, typename ...Types>void copyHostToDevice(memory_unit<DATATYPE> *first)
	{

		size_t datasize = first->getMemorySize();

		cudaError test = cudaSuccess;
		switch (first->getScope())
		{
		case memory_scope::HOST_ONLY:
			break;

		case memory_scope::DEVICE_ONLY:
			break;

		case memory_scope::PINNED:
			test = cudaMemcpy(first->data_device, first->data_host, datasize, cudaMemcpyHostToDevice);
			break;

		case memory_scope::NON_PINNED:
			test =cudaMemcpy(first->data_device, first->data_host, datasize, cudaMemcpyHostToDevice);
			break;
		}

		if (test != cudaSuccess)
		{
			std::cout << "Error copying memory to device.";
		}

	}

	template<typename DATATYPE, typename ...Types>void initiateVaribles(memory_unit<DATATYPE> *first, Types* ... rest)
	{
		initiateVaribles(first);
		initiateVaribles(rest...);
	}

	template<typename DATATYPE> void initiateVaribles(memory_unit<DATATYPE> *first)
	{

		size_t datasize = first->getMemorySize();

		cudaError test = cudaSuccess;
		switch (first->getScope())
		{
		case memory_scope::HOST_ONLY:
			first->data_host = (DATATYPE*)malloc(datasize);
			break;

		case memory_scope::DEVICE_ONLY:
			test = cudaMalloc((void **)&first->data_device, datasize);
			break;

		case memory_scope::PINNED:
			test = cudaMallocHost((void**)&first->data_host, datasize);
			cudaMalloc((void **)&first->data_device, datasize);
			break;

		case memory_scope::NON_PINNED:
			test = cudaMalloc((void **)&first->data_device, datasize);


			first->data_host = (DATATYPE*)malloc(datasize);
			break;

			//case MemoryScope::TEXTURE:
			//	first->data_host = (DATATYPE*)malloc(datasize);
			//	break;
		}

		if (test != cudaSuccess)
		{
			std::cout << "Error allocating memory.";
		}
	}

}
#endif
