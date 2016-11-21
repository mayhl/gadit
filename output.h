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

#ifndef OUTPUT
#define OUTPUT

#include <fstream>
#include <iostream>
#include <string>

namespace file_directories
{
		std::string tmpDir				= "temp";
		std::string statusDir			= "status";
		std::string inputDir			= "input";
		std::string outputDir			= "output";

		
		std::string backupSolution		= tmpDir	+ "/BackupSolutionData.bin";
		std::string backupFileInfo		= tmpDir	+ "/BackupInfo.bin";

		std::string icInputFile			= inputDir	+ "/InitialCondition.bin";
		
		std::string timestepData		= statusDir + "/Timesteps.bin";
		std::string newtonCountData		= statusDir + "/NewtonIterationCounts.bin";
		std::string gaditStatus 		= statusDir + "/GADITstatus.bin";
		std::string gaditStatusIndices	= statusDir + "/GADITstatusIndices.bin";

		std::string statusData			= statusDir + "/Status.txt";

		void clean()
		{
			remove(backupSolution.c_str());
			remove(backupFileInfo.c_str());
			remove(timestepData.c_str());
			remove(newtonCountData.c_str());
			remove(gaditStatus.c_str());
			remove(gaditStatusIndices.c_str());
			remove(statusData.c_str());
		}

}


using namespace std;

bool fileExists(const char* inputFile) {
  ifstream ifile(inputFile);
  
  if (ifile)
  {
	  ifile.close();
	  return true;
	}
  else 
	  return false;
};

bool fileExists(const std::string& ifile) {
  return fileExists(ifile.c_str());
};


template <typename DATATYPE> void save_object( const std::string& filename , DATATYPE  object )
{
	ofstream output(filename.c_str(), ios::out | ios::binary);
	output.write((char *) &object, sizeof(DATATYPE));
	output.close();
}

template <typename DATATYPE> void load_object( const std::string& filename , DATATYPE  &object )
{
	ifstream input(filename.c_str(), ios::in | ios::binary);
	input.read((char *) &object, sizeof(DATATYPE));
	input.close();
};


void write_to_old_file( const std::string& filename , const  std::string&  text )
{
	ofstream output(filename.c_str(),ios::app );
	output << text;
	output << "\n";
	output.close();

	
	printf( text.c_str() );
	printf( "\n");
}

void write_to_new_file(const std::string& filename , const  std::string&  text )
{
	ofstream output(filename.c_str() , ios::binary );
	output << text;
	output << "\n";
	output.close();

	
	printf( text.c_str() );
	printf( "\n");
}

template <typename DATATYPE> void load_binary( const std::string& filename , DATATYPE *data, int n, int m)
{
	ifstream input(filename.c_str(), ios::in | ios::binary);
	input.read((char *) data, n*m*sizeof(DATATYPE));
	input.close();
};

template <typename DATATYPE> void output_append_vector( const std::string& filename , DATATYPE *data, int n )
{
	ofstream output(filename.c_str(),ios::app | ios::binary  );
	output.write((char *) data, n*sizeof(DATATYPE));
	output.close();
}

template <typename DATATYPE> void output_binary( const std::string& filename , DATATYPE *data, int n, int m)
{
	ofstream output(filename.c_str(), ios::out | ios::binary);
	output.write((char *) data, n*m*sizeof(DATATYPE));
	output.close();
}
#endif