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

#ifndef MODEL_COLLOID_POLYMER
#define MODEL_COLLOID_POLYMER

// Imports base model_parameters template.
#include "solver_template.h"

namespace model_colloid_polymer
{
	// If copying to make new model, change to new model id.
	const model::id ID = model::COLLOID_POLYMER;

	template <typename DATATYPE > struct model_parameters<DATATYPE, ID>
	{

		// Main parameters
		DATATYPE K;
		DATATYPE b;
		DATATYPE c;
		DATATYPE np;
		DATATYPE Gamma;
		DATATYPE phi_star;

		// Derived parameters common to each spacial point,
		// precomputed on CPU to remove redundant GPU computations.
		// Only required if expression for nonlinear functions 
		// are complicated and contain many fixed terms.
		DATATYPE GammaK;
		DATATYPE b2;
		DATATYPE b4;
		//DATATYPE beta2;
		//DATATYPE w2;
		//DATATYPE two_w;
		//DATATYPE three_b;
		//DATATYPE scaled_cN;
		//DATATYPE cK_b2;
		//DATATYPE scaled_cK_b3;

		std::string to_string()
		{
			std::string output;
			output = format_parameter_output::make_title("Colloid Polymer Model");

			//output += "cK   = " + format_parameter_output::datatype(this->cK) + "\n";
			//output += "cC   = " + format_parameter_output::datatype(this->cC) + "\n";
			//output += "cN   = " + format_parameter_output::datatype(this->cN) + "\n";
			//output += "beta = " + format_parameter_output::datatype(this->beta) + "\n";
			//output += "b    = " + format_parameter_output::datatype(this->b) + "\n";
			//output += "w    = " + format_parameter_output::datatype(this->w) + "\n";
			output += "\n";

			return output;
		}

		void compute_derived_parameters()
		{
			GammaK = Gamma*K;
			b2 = b*b;
			b4 = b2*b2;
			//inv_w = 1 / w;;
			//beta2 = beta*beta;
			//w2 = w*w;
			//three_b = 3 * b;
			//two_w = 2 * w;
			//scaled_cN = 0.25*cN*inv_w;
			//cK_b2 = cK*b*b;
			//scaled_cK_b3 = 3.0*cK*b*b*b;
		}

		DATATYPE getGrowthRate(DATATYPE h, DATATYPE q)
		{
			//DATATYPE gRate;
			//DATATYPE f1;
			//DATATYPE f2;

			//DATATYPE h2 = h*h;
			//DATATYPE h3 = h2*h;

			//DATATYPE tau = tanh((h - 2 * b)*inv_w);
			//DATATYPE kappa = tau - 1;;

			//DATATYPE eta = h2 + beta*beta;
			//DATATYPE eta2 = eta*eta;
			//DATATYPE invEta = 1.0 / eta;

			//DATATYPE shareNematic = scaled_cN*(tau + 1)*(tau + 1)*invEta*invEta*invEta*h3;
			//DATATYPE invH = 1.0 / h;

			//f2 = -h*shareNematic*(h2*two_w - w*eta + h*eta*(tau - 1));
			//f2 += cK_b2*(2 - three_b*invH);

			//f1 = cC*h3;

			//DATATYPE q2 = q*q;
			//gRate = -(f1*q2 - f2)*q2;

			return 0;// gRate;


		}
		DATATYPE getMaxGrowthMode(DATATYPE h)
		{

	


			DATATYPE f1 = GammaK;

			DATATYPE phi = h;
			DATATYPE A = phi - phi_star;
			DATATYPE A2 = A*A;
			DATATYPE B = c - phi;
			DATATYPE B2 = B*B;
			DATATYPE C = 1 + b2*B*B;
			DATATYPE C2 = C*C;
			DATATYPE C3 = C*C2;


			DATATYPE dfc = 8 * b4*B2 / (A*C*C2) - 2 * b2 / (A*C2) - 4 * b2*B / (A2*C2) + 2 / (A2*A*C) - 2 / (phi*phi*phi);

			DATATYPE d2fc = 48 * b4*b2*B2*B / (A*C2*C2) - 24 * b4*B / (A*C3) - 24 * b4*B2 / (A2*C3) + 6 * b2 / (A2*C2) + 12 * b2*B / (A2*A*C2) - 6 / (A2*A2*C) + 6 / (phi*phi*phi*phi);

			DATATYPE D = 1.0 / (1.0 - phi);

			DATATYPE dfp = -7.0*np*D*D;
			DATATYPE d2fp = 2.0*D*dfp;

			DATATYPE f2 = dfc + dfp;

			//DATATYPE qm;
			//DATATYPE f1;
			//DATATYPE f2;

			//DATATYPE h2 = h*h;
			//DATATYPE h3 = h2*h;

			//DATATYPE tau = tanh((h - 2 * b)*inv_w);
			//DATATYPE kappa = tau - 1;;

			//DATATYPE eta = h2 + beta*beta;
			//DATATYPE invEta = 1.0 / eta;

			//DATATYPE shareNematic = scaled_cN*(tau + 1)*(tau + 1)*invEta*invEta*invEta*h3;
			//DATATYPE invH = 1.0 / h;

			//f2 = -h*shareNematic*(h2*two_w - w*eta + h*eta*(tau - 1));
			//f2 += cK_b2*(2 - three_b*invH);

			//f1 = cC*h3;

			DATATYPE test = f2 / (2.0*f1);
			DATATYPE qm = sqrt(test);

			return qm;
		}

	};


	// Dummy subroutine template created so switch in subroutine
	// 'newton_iterative_method::compute_nonlinear_functions()'
	// has valid compile time function calls for switch paths 
	// that are not current model ID. 
	// Note: By construction, subroutine is never called at runtime. 
	template <typename DATATYPE, model::id MODEL_ID> __device__ __forceinline__
		void nonlinear_functions
		(reduced_device_workspace<DATATYPE> d_ws, DATATYPE *d_h, model_parameters<DATATYPE, MODEL_ID> modelParas, int h_index);


	// Definition of nonlinear functions for this model ID.
	template <typename DATATYPE, model::id MODEL_ID = ID> __device__ __forceinline__
		void nonlinear_functions(reduced_device_workspace<DATATYPE> d_ws, DATATYPE *d_h, model_parameters<DATATYPE, ID > modelParas, int h_index)
	{
		DATATYPE GammaK = modelParas.GammaK;
		DATATYPE phi_star = modelParas.phi_star;
		DATATYPE c = modelParas.c;
		DATATYPE b = modelParas.b;
		DATATYPE b2 = modelParas.b2;
		DATATYPE b4 = modelParas.b4;
		DATATYPE np = modelParas.np;


		d_ws.f1[h_index] = GammaK;

		d_ws.df1[h_index] = 0;

		DATATYPE phi = d_h[h_index];
		DATATYPE A = phi - phi_star;
		DATATYPE A2 = A*A;
		DATATYPE B = c - phi;
		DATATYPE B2 = B*B;
		DATATYPE C = 1 + b2*B*B;
		DATATYPE C2 = C*C;
		DATATYPE C3 = C*C2;


		DATATYPE dfc = 8 * b4*B2 / (A*C*C2) - 2 * b2 / (A*C2) - 4 * b2*B / (A2*C2) + 2 / (A2*A*C) - 2 / (phi*phi*phi);

		DATATYPE d2fc = 48 * b4*b2*B2*B / (A*C2*C2) - 24 * b4*B / (A*C3) - 24 * b4*B2 / (A2*C3) + 6 * b2 / (A2*C2) + 12 * b2*B / (A2*A*C2) - 6 / (A2*A2*C) + 6 / (phi*phi*phi*phi);

		DATATYPE D = 1.0 / (1.0 - phi);

		DATATYPE dfp = -7.0*np*D*D;
		DATATYPE d2fp = 2.0*D*dfp;

		d_ws.f2[h_index] = dfc + dfp;
		d_ws.df2[h_index] = d2fc + d2fp;



		//DATATYPE dfc = 2 * b2*B / (A*C*C) - 1.0 / (A2*C) + 1 / (phi*phi);

		//DATATYPE d2fc = 8 * b4*B*B / (A*C*C2) - 2 * b2 / (A*C2) - 4 * b2*B / (A2*C2) + 2 / (A2*A*C) - 2 / (phi*phi*phi);

		//DATATYPE D = 1.0 / (1.0 - phi);

		//DATATYPE dfp = -7.0*np*D;
		//DATATYPE d2fp = dfp*D;



		//DATATYPE tau = tanh((h - 2 * b)*invW);
		//DATATYPE kappa = tau - 1;;

		//DATATYPE eta = h2 + beta2;
		//DATATYPE eta2 = eta*eta;
		//DATATYPE invEta = 1.0 / eta;

		//DATATYPE shareNematic = scaled_cN*(tau + 1)*(tau + 1)*invEta*invEta*invEta*h3;
		//DATATYPE invH = 1.0 / h;

		//d_ws.f2[h_index] = -h*shareNematic*(h2*two_w - w*eta + h*eta*kappa);
		//d_ws.f2[h_index] += cK_b2*(2 - three_b*invH);

		//d_ws.df2[h_index] = invW*invEta*shareNematic*(4 * w2*(3 * h2*h2 - 4 * h2*eta + eta2) + h*eta*(8 * h2*w + 2 * h*eta - 7 * w*eta)*kappa + 3 * h2*eta2*kappa*kappa);
		//d_ws.df2[h_index] += scaled_cK_b3*invH*invH;

	}


}

#endif