#pragma once
#include <functional>
#include "dg/backend/memory.h"
#include "dg/topology/geometry.h"
#include "dg/blas1.h"

namespace dg
{

namespace geo
{

///@addtogroup fluxfunctions

//NOTES ON HOW TO USE THE FUNCTORS/STRUCTURES

//When you define a structure (like convolution2), it needs a constructor to be completely defined.
//A constructor is defined with the same name as the structure, and between brackets, the inputs required to pre define the structure.
//If a constructor is not define, there is always a default constructor defined, which is empty, so you need no input and does not make anything.
//The structure of the constructor is as follow:

//name_of_constructor(equal to the structure)(list of inputs):
//initializer list (where we define the constructors to be used on the private variables. ) {}
//There can be several constructors, but with the same name but different number or kind of inputs.
//Inside of the {} calculations can be done, but no return is going to have.
//To create the constructor in the main code it is done like this: Convolution2 conv(  range, f0, gridX);
//conv is the name that is going to have that constructor 
//It is neccesary before using any function inside of the structure.

//After this, we can define functions with the following structure:
//kind of return( int/double/HVec) name_of_function(whatever)(input list) {WHat the function does;
//return output}

//Anything that wants to be used inside of the structure (in functions or constructures) need to be defined in the private section.
// If some of these structures to be used have no default constructor (as averages), it needs to be defined in the initializer list in the constructor
// 


//EXTRA INFO: If we define a function as operator()(inputlist), it means that to call that function, instead of using the name of the constructor already used:
//like conv.name_function(input for function) it simply works like conv(input for function)


//HVecs are only 1 dimensional, so when they are "2D", they are simply 1D, so if we want to work it as Matrices, we have to define things like this:
//conv2d[i*Nzeta+k] = F1D[k]; with i and k the indeces of the two directions of the grid
// like this: std::vector<HVEc> v(10); I can define a vecor of vectors, which would work as a matrix, and could acces the entries as:
//v[0][1] = 1;

struct Convolution_def
{
	//This functor is used to obtain the convolution of quantities from the formula described in feltor.pdf
	// In the initializer, the convolution of the volume matrix is obtained and saved in m_conv_volX2d.
	
	
	Convolution_def(double range, RealCurvilinearGridX2d<double> gridX2d): 
	m_range(range), m_g(gridX2d), m_poloidal_average(m_g.grid(), dg::coo2d::y)
	{	volX2d = dg::tensor::volume2d(m_g.metric()); 
		std::vector<dg::HVec > coordsX = m_g.map();
		dg::blas1::pointwiseDot( coordsX[0], volX2d, volX2d);
		dg::HVec m_conv_vol1d;
		dg::HVec m_volX2d(volX2d);
		double d_eta=(m_g.ly())/(m_g.Ny()*m_g.n());
		
		m_conv_volX2d=volX2d;
		for (double eta=m_g.y0(); eta<m_g.y1(); eta+=d_eta) 
			{ 	dg::blas1::pointwiseDot(volX2d, dg::evaluate(dg::geo::Grid_cutter(eta, m_range), m_g.grid()), m_volX2d);                                             
				m_poloidal_average(m_volX2d , m_conv_vol1d, false);
				dg::blas1::scal( m_conv_vol1d, 2*M_PI);				
				for (unsigned int zeta=0; zeta<m_g.n()*m_g.Nx(); zeta++)
				{m_conv_volX2d[(eta/d_eta)*m_g.n()*m_g.Nx()+zeta]=m_conv_vol1d[zeta];};
			};
		
	}
	
	HVec convoluted_grid() {return m_conv_volX2d;} //Function to return the convoluted volume matrix
	
	HVec convolute(const HVec F){ //Function to make the convolution of F function.
	dg::HVec m_conv_F1d;
	m_F=F;
	double d_eta=m_g.ly()/(m_g.Ny()*m_g.n());
	m_conv_F=F;

	for (double eta=m_g.y0(); eta<m_g.y1(); eta+=d_eta) 
			{ 
				dg::blas1::pointwiseDot(F, dg::evaluate(dg::geo::Grid_cutter(eta, m_range), m_g.grid()), m_F);                                               
				m_poloidal_average(m_F, m_conv_F1d, false);
				dg::blas1::scal( m_conv_F1d, 2*M_PI);
				for (unsigned int zeta=0; zeta<m_g.n()*m_g.Nx(); zeta++)
				{
				m_conv_F[(eta/d_eta)*m_g.n()*m_g.Nx()+zeta]=m_conv_F1d[zeta];  					
				};
			};
		
	return m_conv_F;
	}
	
	HVec radial_cut(const HVec F, const double zeta_def){ //This functions takes a 2D object in the Xgrid plane at a define radial position and saves it in a 1D variable with eta dependence.
	dg::Grid1d g1d_out_eta(m_g.y0(), m_g.y1(), m_g.n(), m_g.Ny(), dg::DIR_NEU); 
	m_conv_LCFS_F=dg::evaluate( dg::zero, g1d_out_eta);
	unsigned int zeta_cut=round(((zeta_def-m_g.x0())/m_g.lx())*m_g.Nx()*m_g.n());

	for (unsigned int eta=0; eta<m_g.n()*m_g.Ny(); eta++) 
	{m_conv_LCFS_F[eta]=F[eta*m_g.n()*m_g.Nx()+zeta_cut];}
	return m_conv_LCFS_F;	
	}
	
	
	private:
	HVec m_F, m_conv_F, m_conv_LCFS_F, volX2d, m_conv_volX2d;
    double m_range;
    RealCurvilinearGridX2d<double> m_g;
	Average<dg::HVec> m_poloidal_average;		

};
}
}


