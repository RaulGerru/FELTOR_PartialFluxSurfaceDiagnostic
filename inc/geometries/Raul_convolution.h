#pragma once
#include <functional>
#include "dg/backend/memory.h"
#include "dg/topology/geometry.h"
#include "dg/blas1.h"

namespace dg
{

namespace geo
{
	
struct convolution : public aCylindricalFunctor<convolution> 
{ 

	convolution(HVec conv_transferH2dX, double range, const double f0, RealCurvilinearGridX2d<double> gridX2d): Conv_transferH2dX(conv_transferH2dX), Range(range), F0(f0), m_g(gridX2d) {}

		double do_compute(double zeta, double eta) const {
		if( zeta<0 &&  zeta>-0.0009999){
			//std::cout << "zeta="<<zeta<<"\n";
         
			dg::Average<dg::HVec > Poloidal_average( m_g.grid(), dg::coo2d::y);			
			dg::SparseTensor<dg::HVec> metricX = m_g.metric();
			dg::HVec in_VolX2d = dg::tensor::volume2d( metricX);
			dg::HVec Cutted_transferH2dX=Conv_transferH2dX;
			dg::HVec Conv_dvdpsip, Conv_t1d;

			dg::HVec cut= dg::evaluate(dg::geo::Grid_cutter2(eta, Range, zeta), m_g.grid());
			dg::blas1::pointwiseDot(in_VolX2d, cut, in_VolX2d); //cut the volume grid to do the partial flux surface integral                                              	
			Poloidal_average(in_VolX2d, Conv_dvdpsip, false);
			dg::blas1::scal(Conv_dvdpsip, 4.*M_PI*M_PI*F0);
			
			dg::blas1::pointwiseDot( Cutted_transferH2dX, in_VolX2d, Cutted_transferH2dX);
			Poloidal_average(Cutted_transferH2dX, Conv_t1d, false);
			dg::blas1::scal(Conv_t1d, 4*M_PI*M_PI*F0); 
			dg::blas1::pointwiseDivide(Conv_t1d, Conv_dvdpsip, Conv_t1d);
			

			double result;
			dg::HVec ones(Conv_t1d);
			dg::blas1::copy( 1., ones);
			int lim=Conv_t1d.size();
			
			for (int i=0; i<lim; i++)
			{ if (Conv_t1d[i]!=Conv_t1d[i]) Conv_t1d[i]=0;}
			
			result=dg::blas1::dot(ones, Conv_t1d);

					/*
			for (int i=0; i<480; i++)
			{ if (Conv_t1d_2[i]!=Conv_t1d_2[i]){}
				else {
				result=Conv_t1d_2[i];
				break;
			}				
			}
			* 
			* 
			* 			//for (int i=0; i<480; i++)
			//{ if (Conv_t1d_2[i]!=Conv_t1d_2[i]) Conv_t1d_2[i]=0;}
			*/
			
			
			return result;
			}
			else
			{//std::cout << "zeta="<<zeta<<"\n";
			return 0;
			}
			}
			
	private:
	HVec Conv_transferH2dX, VolX2d ;
    double Range;
    const double F0;
    RealCurvilinearGridX2d<double> m_g;
	
};
}
}


