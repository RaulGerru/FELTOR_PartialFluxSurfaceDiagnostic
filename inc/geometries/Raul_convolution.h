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
	
	
struct convolution : public aCylindricalFunctor<convolution> 
{ 
	/**
    * @brief It makes a poloidal convolution of the different data points of a 2D X-grid: It cuts the data and makes the poloidal average that corresponds to that point in the grid in a range. 
    *
    * For the moment, it has a cut in the zeta range to make it only in the LCFS, as the computation time is quite expensive.
    * 
    * \f[ f(zeta,eta)= \begin{cases}
	*mean(f(zeta,eta)) from eta-range/2 to eta+range/2 \text{ if } zeta=0 (LCFS) \\
	*0 \text{ else }
	*\end{cases}
	*\f]
    * 
    * 
    * @brief <tt> convolution( data, range, f0, GridX2d) </tt>
    * @tparam HVec
    * @param data (THe data matrix that you want to produce the convolution in)
    * @tparam double
    * @param range (how big the poloidal angle in which you want to produce the convolution, in degrees). This should be related to the repetition of structures in the toroidal average, which must be dependent on the toroidal discretitation, the q factor and the parallel velocity.
    * @tparam RealCurvilinearGridX2d<double>
    * @param GridX2d (THe grid over which the data is plotted and on which you are supposed to do the cut with the Grid_cutter functions)
    * 
    * 
    * 
    * 
    * @note How to use it? dg::evaluate(dg::geo::convolution(transferH2dX, 7.5, f0, gridX2d), gridX2d.grid());  After you have it, you will need to get the data to a simple vector, and not a matrix;
    */

	convolution(HVec conv_transferH2dX, double range, const double f0, RealCurvilinearGridX2d<double> gridX2d): Conv_transferH2dX(conv_transferH2dX), Range(range), F0(f0), m_g(gridX2d), POloidal_average(gridX2d.grid(), dg::coo2d::y) {}

		double do_compute(double zeta, double eta) const {
		if( zeta<0 &&  zeta>-0.0009999){ //For the moment, it has this cut to only look at the LCFS. It might be interesting to go a couple of points inner so we avoid the problems at 0.
			         
			         //ALL of th paragraph can be defined out of the loop
			//dg::Average<dg::HVec > Poloidal_average( m_g.grid(), dg::coo2d::y);	//define poloidal average
			dg::SparseTensor<dg::HVec> metricX = m_g.metric();
			dg::HVec In_VolX2d = dg::tensor::volume2d( metricX);
			dg::HVec in_VolX2d2 = In_VolX2d; //Definition of the variables that we are going to edit inside of the loop
			dg::HVec Cutted_transferH2dX=Conv_transferH2dX;
			dg::HVec Cutted_transferH2dX2=Conv_transferH2dX;
			dg::HVec Conv_dvdpsip, Conv_t1d, Conv_dvdpsip2, Conv_t1d2;



			dg::HVec cut= dg::evaluate(dg::geo::Grid_cutter2(eta, Range, zeta), m_g.grid()); //Cutting the grid with the range and the LCFS (zeta with the if initial condition)
			dg::HVec cut2= dg::evaluate(dg::geo::Grid_cutter2(eta, 2*Range, zeta), m_g.grid()); //This is a try for the double period 
			dg::blas1::pointwiseDot(In_VolX2d, cut, In_VolX2d); //cut the volume grid to do the partial flux surface integral                                              	
			dg::blas1::pointwiseDot(in_VolX2d2, cut2, in_VolX2d2); //cut the volume grid to do the partial flux surface integral                                              				
			POloidal_average(In_VolX2d, Conv_dvdpsip, false); //Make the poloidal average of the cutted volume matrix
			POloidal_average(in_VolX2d2, Conv_dvdpsip2, false);
			dg::blas1::scal(Conv_dvdpsip, 4.*M_PI*M_PI*F0); //Normalized the 1D volume vectors
			dg::blas1::scal(Conv_dvdpsip2, 4.*M_PI*M_PI*F0);
			
			dg::blas1::pointwiseDot( Cutted_transferH2dX, In_VolX2d, Cutted_transferH2dX); //Cut the data matrix 
			dg::blas1::pointwiseDot( Cutted_transferH2dX2, in_VolX2d2, Cutted_transferH2dX2);
			
			POloidal_average(Cutted_transferH2dX, Conv_t1d, false); //Poloidal average of the cutted data
			POloidal_average(Cutted_transferH2dX2, Conv_t1d2, false);
			dg::blas1::scal(Conv_t1d, 4*M_PI*M_PI*F0); //Normalize the 1D data vector
			dg::blas1::scal(Conv_t1d2, 4*M_PI*M_PI*F0);
			dg::blas1::pointwiseDivide(Conv_t1d, Conv_dvdpsip, Conv_t1d); //Divide the data 1D vector with the Volume to make it simple average and not integral.
			dg::blas1::pointwiseDivide(Conv_t1d2, Conv_dvdpsip2, Conv_t1d2);
			dg::blas1::axpby(0.5,Conv_t1d2, 0.5,Conv_t1d); //Average of the range data and the double range data (try)

			double result;
			dg::HVec ones(Conv_t1d); //We make a ones vector to make a dot product with the data vector and, as the entries of the data re going to be 0 except of the LCFS, we will only get the value in that entry.
			dg::blas1::copy( 1., ones);
			int lim=Conv_t1d.size();
			
			for (int i=0; i<lim; i++) //As we have devided by the cutted dvdpsip, we have entries that are NaN, so we need to make them 0 instead of NaN. That's what this loop makes.
			{ if (Conv_t1d[i]!=Conv_t1d[i]) Conv_t1d[i]=0;} 
			
			result=dg::blas1::dot(ones, Conv_t1d); //Multiply the two vectors to obtain the value at LCFS.

			return result;
			}
			else
			{return 0;}
			}
			
	private:
	HVec Conv_transferH2dX, VolX2d ;
    double Range;
    const double F0;
    RealCurvilinearGridX2d<double> m_g;
    Average<dg::HVec > POloidal_average;	//define poloidal average
	//dg::SparseTensor<dg::HVec> metricX = m_g.metric();
	//HVec In_VolX2d = dg::tensor::volume2d(m_g.metric());
	
};
}
}


